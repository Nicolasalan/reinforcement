#!/usr/bin/env python3

import numpy as np
import random
import torch
from collections import deque, namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
     """Fixed-size buffer to store experience tuples."""

     def __init__(self, buffer_size, batch_size, action_size, seed=0):
          """Initialize a ReplayBuffer object.
          Params
          ======
               buffer_size (int): maximum size of buffer
               batch_size (int): size of each training batch
          """

          self.action_size = action_size # dimensão da ação
          self.memory = deque(maxlen=buffer_size)  # memória interna (deque)
          self.batch_size = batch_size # tamanho do lote
          self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"]) # criar uma nova experiência
          self.seed = random.seed(seed) # semente aleatória
     
     def add(self, state, action, reward, next_state, done):
          """Add a new experience to memory."""
          e = self.experience(state, action, reward, next_state, done) # criar uma nova experiência
          self.memory.append(e)        

     def sample(self):
          """Prioritized experience replay experience sampling."""

          experiences = random.sample(self.memory, k=self.batch_size)

          batch_state = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device) 
          batch_action = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device) 
          batch_rewards = (torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)).reshape(-1, 1)
          batch_next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device) 
          batch_dones = (torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).to(device)).reshape(-1, 1)

          return batch_state, batch_action, batch_rewards, batch_next_states, batch_dones

     def erase(self):
          """Erase the memory."""
          self.memory = self.memory.clear()
          self.count = 0

     def __len__(self):
          """Return the current size of internal memory."""
          return len(self.memory)
     