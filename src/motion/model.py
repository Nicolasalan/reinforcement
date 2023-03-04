#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, seed, l1=800, l2=600):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.layer_1 = nn.Linear(state_dim, l1)
        self.layer_2 = nn.Linear(l1, l2)
        self.layer_3 = nn.Linear(l2, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, seed, l1=800, l2=600):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer_1 = nn.Linear(state_dim, l1)
        self.layer_2_s = nn.Linear(l1, l2)
        self.layer_2_a = nn.Linear(action_dim, l2)
        self.layer_3 = nn.Linear(l2, 1)

        self.layer_4 = nn.Linear(state_dim, l1)
        self.layer_5_s = nn.Linear(l1, l2)
        self.layer_5_a = nn.Linear(action_dim, l2)
        self.layer_6 = nn.Linear(l2, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


