#!/usr/bin/env python3

import numpy as np
import random
import torch
from collections import namedtuple, deque


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
          experience = (state, action, reward, next_state, done)
          if self.num_experiences < self.batch_size:
               self.memory.append(experience)
          else:
               self.memory.popleft()
               self.memory.append(experience)

     def sample(self):
          """Randomly sample a batch of experiences from memory."""
          experiences = random.sample(self.memory, k=self.batch_size)

          states = np.asarray([data[0] for data in experiences])
          actions = np.asarray([data[1] for data in experiences])
          rewards = np.asarray([data[2] for data in experiences])
          next_states = np.asarray([data[3] for data in experiences])
          dones = np.asarray([data[4] for data in experiences])

          return (states, actions, rewards, next_states, dones)

     def erase(self):
          """Erase the memory."""
          self.memory = deque(maxlen=self.buffer_size)

     def __len__(self):
          """Return the current size of internal memory."""
          return len(self.memory)