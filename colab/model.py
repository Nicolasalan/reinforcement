import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Inicializar parâmetros e construir modelo.
        Parâmetros
        ======
            state_size (int): Dimensão de cada estado
            action_size (int): Dimensão de cada ação
            seed (int): Semente aleatória
            fc1_units (int): Número de nós na primeira camada oculta
            fc2_units (int): Número de nós na segunda camada oculta
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed) # define a semente aleatória manualmente
        self.fc1 = nn.Linear(state_size, fc1_units) # camada de entrda com 400 nós e estado de tamanho 33
        self.bn1 = nn.BatchNorm1d(fc1_units) # normalização da camada de entrada para evitar overfitting
        self.fc2 = nn.Linear(fc1_units, fc2_units) # camada oculta com 300 nós e 400 de entrada
        self.fc3 = nn.Linear(fc2_units, action_size) # camada de saída com 4 nós e 300 de entrada
        self.reset_parameters() # resetar os pesos da rede

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1)) # inicializar os pesos da camada de entrada com valores aleatórios
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2)) # inicializar os pesos da camada oculta com valores aleatórios
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Construa uma rede de atores (políticas) que mapeie estados -> ações."""
        x = F.relu(self.bn1(self.fc1(state))) # aplicar a função de ativação relu na camada de entrada
        x = F.relu(self.fc2(x)) # aplicar a função de ativação relu na camada oculta
        return F.tanh(self.fc3(x)) # aplicar a função de ativação tanh na camada de saída


class Critic(nn.Module):
    """Modelo Crítico (Valor)."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Inicializar parâmetros e construir modelo.
        Parâmetros
        ======
            state_size (int): Dimensão de cada estado
            action_size (int): Dimensão de cada ação
            seed (int): Semente aleatória
            fcs1_units (int): Número de nós na primeira camada oculta
            fc2_units (int): Número de nós na segunda camada oculta
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Construa uma rede crítica (valor) que mapeia pares (estado, ação) -> valores-Q."""
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1) # concatenar a camada de entrada com a camada de ação
        x = F.relu(self.fc2(x))
        return self.fc3(x)