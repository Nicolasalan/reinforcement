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
          self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
          self.batch_size = batch_size
          self.seed = random.seed(seed)
     
     def add(self, state, action, reward, next_state, done):
          """Add a new experience to memory."""
          experience = (state, action, reward, next_state, done) # criar uma nova experiência
          self.memory.append(experience) 

     def sample(self):
          """Randomly sample a batch of experiences from memory."""
          experiences = random.sample(self.memory, k=self.batch_size)

          states = torch.Tensor([data[0] for data in experiences]).to(device)
          actions = torch.Tensor([data[1] for data in experiences]).to(device)
          rewards = torch.Tensor([data[2] for data in experiences]).to(device).reshape(-1, 1)
          next_states = torch.Tensor([data[3] for data in experiences]).to(device).reshape(-1, 1)
          dones = torch.Tensor([data[4] for data in experiences]).to(device)

          return (states, actions, rewards, next_states, dones)

     def erase(self):
          """Erase the memory."""
          self.memory = deque(maxlen=self.batch_size)

     def __len__(self):
          """Return the current size of internal memory."""
          return len(self.memory)