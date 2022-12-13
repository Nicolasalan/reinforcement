#!/usr/bin/env python3

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_dim, action_dim, seed, fc1_units=800, fc2_units=600):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()

        print(state_dim, action_dim)
        self.state_input = torch.Size([None, state_dim])

        self.seed = torch.manual_seed(seed)
        self.layer_1 = nn.Linear(state_dim, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.layer_2 = nn.Linear(fc1_units, fc2_units)
        self.layer_3 = nn.Linear(fc2_units, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        state = F.relu(self.state_input(state))
        state = F.relu(self.bn1(self.layer_1(state)))
        state = F.relu(self.layer_2(state))
        a = self.tanh(self.layer_3(state))
        return a

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_dim, action_dim, seed, fc1_units=800, fc2_units=600):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer_1 = nn.Linear(state_dim, fc1_units)
        self.layer_2_s = nn.Linear(fc1_units, fc2_units)
        self.layer_2_a = nn.Linear(action_dim, fc2_units)
        self.layer_3 = nn.Linear(fc2_units, 1)

        self.seed = torch.manual_seed(seed)
        self.layer_4 = nn.Linear(state_dim, fc1_units)
        self.layer_5_s = nn.Linear(fc1_units, fc2_units)
        self.layer_5_a = nn.Linear(action_dim, fc2_units)
        self.layer_6 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        s1 = F.relu(self.bn1(self.layer_1(state)))
        self.layer_2_s(s1)
        self.layer_2_a(action)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(action, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(state))
        self.layer_5_s(s2)
        self.layer_5_a(action)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(action, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2