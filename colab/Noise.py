import numpy as np
import random
import copy

class OUNoise:
     """Processo Ornstein-Uhlenbeck."""

     def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
          """Inicializar parâmetros e processo de ruído."""
          self.mu = mu * np.ones(size) # adicionar ruído
          self.theta = theta # taxa de decaimento 
          self.sigma = sigma # taxa de variação
          self.seed = random.seed(seed) # semente aleatória
          self.reset() # redefinir o estado interno

     def reset(self):
          """Redefinir o estado interno (= ruído) para significar (mu)."""
          self.state = copy.copy(self.mu) # copiar o ruído

     def sample(self):
          """Atualizar o estado interno e retorná-lo como uma amostra de ruído."""
          x = self.state # estado interno
          dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))]) # atualizar o estado interno 
          self.state = x + dx # atualizar o estado interno 
          return self.state # retornar o estado interno