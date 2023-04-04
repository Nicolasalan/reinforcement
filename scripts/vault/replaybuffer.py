#!/usr/bin/env python3

import numpy as np
import random
import torch
from collections import deque
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from typing import Iterator, List, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
     write = 0

     def __init__(self, capacity):
          self.capacity = capacity
          self.tree = np.zeros(2 * capacity - 1)
          self.data = np.zeros(capacity, dtype=object)
          self.n_entries = 0

     # update to the root node
     def _propagate(self, idx, change):
          parent = (idx - 1) // 2

          self.tree[parent] += change

          if parent != 0:
               self._propagate(parent, change)

     # find sample on leaf node
     def _retrieve(self, idx, s):
          left = 2 * idx + 1
          right = left + 1

          if left >= len(self.tree):
               return idx

          if s <= self.tree[left]:
               return self._retrieve(left, s)
          else:
               return self._retrieve(right, s - self.tree[left])

     def total(self):
          return self.tree[0]

     # store priority and sample
     def add(self, p, data):
          idx = self.write + self.capacity - 1

          self.data[self.write] = data
          self.update(idx, p)

          self.write += 1
          if self.write >= self.capacity:
               self.write = 0

          if self.n_entries < self.capacity:
               self.n_entries += 1

     # update priority
     def update(self, idx, p):
          change = p - self.tree[idx]

          self.tree[idx] = p
          self._propagate(idx, change)

     # get priority and sample
     def get(self, s):
          idx = self._retrieve(0, s)
          dataIdx = idx - self.capacity + 1

          return (idx, self.tree[idx], self.data[dataIdx])
    
class ReplayBuffer:
     """Fixed-size buffer to store experience tuples."""

     def __init__(self, capacity: int) -> None:
          """Initialize a ReplayBuffer object.
          Params
          ======
               buffer_size (int): maximum size of buffer
               batch_size (int): size of each training batch
          """
          self.tree = SumTree(capacity)
          self.capacity = capacity
          self.e = 0.01 
          self.a = 0.6
          self.beta = 0.4
          self.beta_increment_per_sampling = 0.00001

     def _get_priority(self, error):
          return (np.abs(error) + self.e) ** self.a
     
     def add(self, error, sample) -> None:
          """Add experience to the buffer.

          Args:
               experience: tuple (state, action, reward, done, new_state)
          """
          p = self._get_priority(error)
          self.tree.add(p, sample)      

     def sample(self, n) -> tuple:
          """Prioritized experience replay experience sampling."""
          batch = []
          idxs = []
          segment = self.tree.total() / n
          priorities = []

          self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

          for i in range(n):
               a = segment * i
               b = segment * (i + 1)

               s = random.uniform(a, b)
               (idx, p, data) = self.tree.get(s)
               priorities.append(p)
               batch.append(data)
               idxs.append(idx)

          sampling_probabilities = priorities / self.tree.total()
        
          weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
          
          weight /= weight.max()

          return batch, idxs, weight

     def update(self, idx, error) -> None:
          p = self._get_priority(error)
          self.tree.update(idx, p)
     

class RLDataset(IterableDataset):
     """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training."""

     def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
          """
          Args:
               buffer: replay buffer
               sample_size: number of experiences to sample at a time
          """
          self.buffer = buffer
          self.sample_size = sample_size

     def __iter__(self) -> Iterator:
          states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
          for i in range(len(dones)):
               yield states[i], actions[i], rewards[i], dones[i], new_states[i]
