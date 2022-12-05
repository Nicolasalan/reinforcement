#!/usr/bin/env python3

import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from agent import Agent

import rospy
import gym
import gym_gazebo
import numpy as np

state_dim = 16
action_dim = 2
action_linear_max = 0.25  # m/s
action_angular_max = 0.5  # rad/s

print('State Dimensions: ' + str(state_dim))
print('Action Dimensions: ' + str(action_dim))
print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')

def main():
     rospy.init_node('ddpg_stage')
     env = Env(is_training)
     agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=42)
