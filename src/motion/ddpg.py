#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from agent import Agent
from environment import Env
from utils import Extension

import rospy
import numpy as np
import os
import yaml

# folder to load config file
CONFIG_PATH = "/ws/src/motion/config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        param = yaml.safe_load(file)

    return param

param = load_config("main_config.yaml")

state_dim = param["environment_dim"] + param["robot_dim"]
action_dim = param["action_dim"]
action_linear_max = param["action_linear_max"]
action_angular_max = param["action_angular_max"]

log = Extension()

def ddpg(n_episodes, print_every, max_t, score_solved):
     print('Starting DDPG ...')
     """
     parameters
     ======
          n_episodes (int): maximum number of training episodes
          max_t(int): maximum number of timesteps per episode
     """
     log.view_parameter()
     agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=42)
     env = Env()

     scores_window = []                                               # average scores of the most recent episodes
     scores = []                                                      # list of average scores of each episode                               

     for i_episode in range(1, n_episodes+1):                         # initialize score for each agent
          rospy.loginfo('Episode: ' + str(i_episode))

          score = 0.0                
          done = False

          agent.reset()                                               # reset environment    
          states = env.reset_env()                                    # get the current state of each agent

          for t in range(max_t):                            
               action = agent.action(states)                          # choose an action for each agent
               actions = [(action[0] + 1) / 2, action[1]]

               next_states, rewards, dones, _ = env.step_env(actions) # send all actions to the environment

               # save the experiment in the replay buffer, run the learning step at a defined interval
               agent.step(states, actions, rewards, next_states, dones, t)
          
               states = next_states
               score += rewards
               if np.any(done):                                       # exit loop when episode ends
                    break              
               
               scores_window.append(score)                            # save average score for the episode
               scores.append(score)                                   # save average score in the window
               
          if i_episode % print_every == 0:
               rospy.logwarn('# ====== Episode: ' + str(i_episode) + ' Average Score: ' + str(np.mean(scores_window)) + ' ====== #')
          if np.mean(scores_window) >= score_solved:
               rospy.logwarn('Environment solved in ' + str(i_episode) + ' episodes!' + ' Average Score: ' + str(np.mean(scores_window)))
               torch.save(agent.actor_local.state_dict(), 'actor_checkpoint.pth')
               torch.save(agent.critic_local.state_dict(), 'critic_checkpoint.pth')
               break

     return scores

if __name__ == '__main__':
     """Start training."""
     
     n_episodes = param["N_EPISODES"]
     print_every = param["PRINT_EVERY"] 
     max_t = param["MAX_T"]
     score_solved = param["SCORE_SOLVED"]

     ddpg(n_episodes, print_every, max_t, score_solved)