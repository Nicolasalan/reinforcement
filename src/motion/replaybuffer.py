#!/usr/bin/env python3

import numpy as np
import random
import torch
from collections import deque

# importar utilitarios
from utils import Extension

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
          self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
          self.batch_size = batch_size
          self.seed = random.seed(seed)
          self.util = Extension()
     
     def add(self, state, action, reward, next_state, done):
          """Add a new experience to memory."""
          experience = (state, action, reward, next_state, done) # criar uma nova experiência
          self.memory.append(experience) 

     def sample(self):
          """Randomly sample a batch of experiences from memory."""
          batch = []

          batch = random.sample(self.memory, k=self.batch_size)
        
          batch_state       = np.array([_[0] for _ in batch])
          batch_action      = np.array([_[1] for _ in batch])
          batch_rewards     = np.array([_[2] for _ in batch]).reshape(-1, 1)
          batch_next_states = np.array([_[3] for _ in batch])
          batch_dones       = np.array([_[4] for _ in batch]).reshape(-1, 1)

          states = torch.Tensor(batch_state).to(device)
          actions = torch.Tensor(batch_action).to(device)
          rewards = torch.Tensor(batch_rewards).to(device)
          next_states = torch.Tensor(batch_next_states).to(device)
          dones = torch.Tensor(batch_dones).to(device)

          return (states, actions, rewards, next_states, dones)

     def erase(self):
          """Erase the memory."""
          self.memory = self.memory.clear()

     def __len__(self):
          """Return the current size of internal memory."""
          return len(self.memory)