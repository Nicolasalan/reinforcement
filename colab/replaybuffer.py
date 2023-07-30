import numpy as np
import random
import torch
from collections import namedtuple, deque


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
     """Buffer de tamanho fixo para armazenar tuplas de experiência."""

     def __init__(self, action_size, buffer_size, batch_size, seed):
          """Inicializar um objeto ReplayBuffer.
          Parâmetros
          ======
               buffer_size (int): tamanho máximo do buffer
               batch_size (int): tamanho de cada lote de treinamento
          """
          self.action_size = action_size # dimensão da ação
          self.memory = deque(maxlen=buffer_size)  # memória interna (deque)
          self.batch_size = batch_size # tamanho do lote
          self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"]) # criar uma nova experiência
          self.seed = random.seed(seed) # semente aleatória
     
     def add(self, state, action, reward, next_state, done):
          """Adicione uma nova experiência à memória."""
          e = self.experience(state, action, reward, next_state, done) # criar uma nova experiência
          self.memory.append(e) # 
     
     def sample(self):
          """Adicione uma nova experiência à memória."""
          experiences = random.sample(self.memory, k=self.batch_size)

          states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device) # converter para tensor
          actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device) # converter para tensor
          rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device) # converter para tensor
          next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device) # converter para tensor
          dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device) # converter para tensor

          return (states, actions, rewards, next_states, dones) # retornar as experiências

     def __len__(self):
          """Retorna o tamanho atual da memória interna."""
          return len(self.memory) # retornar o tamanho da memória interna