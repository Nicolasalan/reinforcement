#!/usr/bin/env python3

import numpy as np
import random
import torch
from collections import deque

from utils import SumTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
     """Fixed-size buffer to store experience tuples."""

     def __init__(self, buffer_size, batch_size, seed, alpha=0.6):
          """Initialize a ReplayBuffer object.
          Params
          ======
               buffer_size (int): maximum size of buffer
               batch_size (int): size of each training batch
          """
          self.memory = deque(maxlen=buffer_size)
          self.batch_size = batch_size
          self.seed = random.seed(seed)
          self.sum_tree = SumTree(buffer_size)
          self.alpha = alpha
          self.max_priority = 1.0
     
     def add(self, state, action, reward, next_state, done):
          """Add a new experience to memory."""
          experience = (state, action, reward, next_state, done)
          self.memory.append(experience) 
          self.sum_tree.add(self.max_priority, experience)
          self.max_priority = max(self.max_priority, self.max_priority * self.alpha)

     def sample(self):
          """Prioritized experience replay experience sampling."""
          batch = []
          priorities = []
          priorities_sum = self.sum_tree.total()
          for i in range(self.batch_size):
               priority = random.uniform(0, priorities_sum)
               idx, data, priority = self.sum_tree.get(priority)
               batch.append(data)
               priorities.append(priority)

          # Normalize the sample priorities
          priorities = np.array(priorities)
          priorities = priorities / priorities_sum
          priorities = priorities ** (1.0 / self.alpha)
          priorities = priorities / np.max(priorities)
        
          # Convert the batch to a numpy array
          batch_state       = np.array([_[0] for _ in batch])
          batch_action      = np.array([_[1] for _ in batch])
          batch_rewards     = np.array([_[2] for _ in batch]).reshape(-1, 1)
          batch_next_states = np.array([_[3] for _ in batch])
          batch_dones       = np.array([_[4] for _ in batch]).reshape(-1, 1)

          # Convert the batch to a torch tensor
          states = torch.Tensor(batch_state).to(device)
          actions = torch.Tensor(batch_action).to(device)
          rewards = torch.Tensor(batch_rewards).to(device)
          next_states = torch.Tensor(batch_next_states).to(device)
          dones = torch.Tensor(batch_dones).to(device)

          return (states, actions, rewards, next_states, dones), priorities, [idx]

     def update_priorities(self, idxs, priorities):
          """Update the priorities of the samples at the given idxs."""
          for i in range(len(idxs)):
               idx = idxs[i]
               priority = priorities[i]
               self.sum_tree.update(idx, priority)
     
     def erase(self):
          """Erase the memory."""
          self.memory = self.memory.clear()

     def __len__(self):
          """Return the current size of internal memory."""
          return len(self.memory)