#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import *

# class Actor(nn.Module):
#     def __init__(self, state_size, action_size, max_action=1, fc1_units=800, fc2_units=600):

#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)
#         self.max_action = max_action

#     def forward(self, state):
#         """Build an actor (policy) network that maps states -> actions."""
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.max_action * torch.tanh(self.fc3(x))

class Actor(nn.Module):
    
    def __init__(self, state_dim = 24):
        super(Actor, self).__init__()
        self.cat_len = 128
        self.CFC = nn.ModuleList()
        self.CFC += [nn.Linear(state_dim, 3*self.cat_len), nn.ReLU(), nn.Dropout(), 
                    nn.Linear(3*self.cat_len, 2), nn.Tanh()]

    def forward(self, state):

        cur_input = state                                                            
        for layer in self.CFC: cur_input = layer(cur_input)                             
        action = cur_input
        
        return action

class Critic(nn.Module):

    def __init__(self, state_dim = 24, action_dim = 2):
        super(Critic, self).__init__()
        self.cat_len = 128

        self.SA = nn.ModuleList()
        self.SA += [nn.Linear(state_dim + action_dim, 3*self.cat_len), nn.ReLU(),
                    nn.Linear(3*self.cat_len, 64),nn.ReLU(), nn.Linear(64 , 1), nn.Identity()]

        self._SA = nn.ModuleList()
        self._SA += [nn.Linear(state_dim + action_dim, 3*self.cat_len), nn.ReLU(),
                    nn.Linear(3*self.cat_len, 64),nn.ReLU(), nn.Linear(64, 1), nn.Identity()]



    def forward(self, state, action):
        
        sa = torch.cat([state, action], dim = -1)
        _sa = torch.cat([state, action], dim = -1)

        for layer in self.SA: sa = layer(sa)
        for layer in self._SA: _sa = layer(_sa)

        final = sa
        _final = _sa

        q1 = torch.squeeze(final, -1)
        q2 = torch.squeeze(_final, -1)
        
        return q1, q2


# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, l1=800, l2=600):
#         super(Critic, self).__init__()
        
#         self.layer_1 = nn.Linear(state_dim + action_dim, l1)
#         self.layer_2_s = nn.Linear(l1, l2)
#         self.layer_2_a = nn.Linear(action_dim, l2)
#         self.layer_3 = nn.Linear(l2, 1)

#         self.layer_4 = nn.Linear(state_dim + action_dim, l1)
#         self.layer_5_s = nn.Linear(l1, l2)
#         self.layer_5_a = nn.Linear(action_dim, l2)
#         self.layer_6 = nn.Linear(l2, 1)

#     def forward(self, s, a):
#         s1 = F.relu(self.layer_1(torch.cat([s, a], dim=1)))
#         s1 = F.relu(self.layer_2_s(s1))
#         a1 = F.relu(self.layer_2_a(a))
#         s1 = s1 + a1
#         q1 = self.layer_3(s1)

#         # Segunda rede crítica
#         s2 = F.relu(self.layer_4(torch.cat([s, a], dim=1)))
#         s2 = F.relu(self.layer_5_s(s2))
#         a2 = F.relu(self.layer_5_a(a))
#         s2 = s2 + a2
#         q2 = self.layer_6(s2)

#         return q1, q2


