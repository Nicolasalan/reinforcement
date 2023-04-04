#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, seed, l1=800, l2=600):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.layer_1 = nn.Linear(state_dim, l1)
        self.norm1 = nn.BatchNorm1d(l1)
        self.layer_2 = nn.Linear(l1, l2)
        self.norm2 = nn.BatchNorm1d(l2)
        self.layer_3 = nn.Linear(l2, action_dim)
        self.dropout = nn.Dropout(0.2)
        self.tanh = nn.Tanh()
        self.reset_parameters()
    
    def reset_parameters(self):
        self.layer_1.weight.data.uniform_(*hidden_init(self.layer_1))
        self.layer_2.weight.data.uniform_(*hidden_init(self.layer_2)) 
        self.layer_3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        state = F.relu(self.layer_1(state))
        state = F.relu(self.layer_2(state))
        action = self.tanh(self.layer_3(state))
        return action

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, seed: int, l1=800, l2=600):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer_1 = nn.Linear(state_dim, l1)
        self.norm1 = nn.BatchNorm1d(l1)
        self.layer_2_s = nn.Linear(l1, l2)
        self.layer_2_a = nn.Linear(action_dim, l2)
        self.dropout1 = nn.Dropout(0.2)
        self.layer_3 = nn.Linear(l2, 1)
        self.reset_parameters_q1()

        self.layer_4 = nn.Linear(state_dim, l1)
        self.norm2 = nn.BatchNorm1d(l1)
        self.layer_5_s = nn.Linear(l1, l2)
        self.layer_5_a = nn.Linear(action_dim, l2)
        self.dropout2 = nn.Dropout(0.2)
        self.layer_6 = nn.Linear(l2, 1)
        self.reset_parameters_q2()

    def reset_parameters_q1(self):
        self.layer_1.weight.data.uniform_(*hidden_init(self.layer_1)) 
        self.layer_2_s.weight.data.uniform_(*hidden_init(self.layer_2_s)) 
        self.layer_2_a.weight.data.uniform_(*hidden_init(self.layer_2_a)) 
        self.layer_3.weight.data.uniform_(-3e-3, 3e-3)

    def reset_parameters_q2(self):
        self.layer_4.weight.data.uniform_(*hidden_init(self.layer_4)) 
        self.layer_5_s.weight.data.uniform_(*hidden_init(self.layer_5_s)) 
        self.layer_5_a.weight.data.uniform_(*hidden_init(self.layer_5_a)) 
        self.layer_6.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        # first critic
        s1 = F.relu(self.layer_1(state))

        # normalize and dropout
        self.norm1(s1)  
        self.dropout1(s1)  
        self.layer_2_s(s1)
        self.layer_2_a(action)

        # multiply with weights and add bias
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(action, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        # second critic
        s2 = F.relu(self.layer_4(state))

        # normalize and dropout
        self.norm2(s2)  
        self.dropout2(s2)
        self.layer_5_s(s2)
        self.layer_5_a(action)

        # multiply with weights and add bias
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(action, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2

