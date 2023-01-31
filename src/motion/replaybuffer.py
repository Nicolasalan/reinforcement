#!/usr/bin/env python3

import numpy as np
import random
import torch
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
     """Fixed-size buffer to store experience tuples."""

     def __init__(self, buffer_size, batch_size, seed):
          """Initialize a ReplayBuffer object.
          Params
          ======
               buffer_size (int): maximum size of buffer
               batch_size (int): size of each training batch
          """
          self.memory = deque(maxlen=buffer_size)
          self.batch_size = batch_size
          self.seed = random.seed(seed)
          self.n = 4
     
     def add(self, state, action, reward, next_state, done):
          """Add a new experience to memory."""
          n_step_return = reward
          for i in range(1, self.n+1):
               if i < len(self.memory):
                    _, _, r, _, _ = self.memory[-i]
                    n_step_return += r
               
          experience = (state, action, reward, next_state, done)
          self.memory.append(experience) 

     def sample(self):
          """Prioritized experience replay experience sampling."""

          # Sample a batch of experiences
          batch = random.sample(self.memory, k=self.batch_size)

          # Convert the batch to a numpy array
          batch_state          = np.array([_[0] for _ in batch])
          batch_action         = np.array([_[1] for _ in batch])
          batch_rewards        = np.array([_[2] for _ in batch]).reshape(-1, 1)
          batch_next_states    = np.array([_[3] for _ in batch])
          batch_n_step_returns = np.array([_[4] for _ in batch]).reshape(-1, 1)
          batch_dones          = np.array([_[4] for _ in batch]).reshape(-1, 1)

          # Convert the batch to a torch tensor
          states = torch.Tensor(batch_state).to(device)
          actions = torch.Tensor(batch_action).to(device)
          rewards = torch.Tensor(batch_rewards).to(device)
          next_states = torch.Tensor(batch_next_states).to(device)
          n_step_returns = torch.Tensor(batch_n_step_returns).to(device)
          dones = torch.Tensor(batch_dones).to(device)

          return (states, actions, rewards, next_states, n_step_returns, dones)

     def erase(self):
          """Erase the memory."""
          self.memory = self.memory.clear()

     def __len__(self):
          """Return the current size of internal memory."""
          return len(self.memory)