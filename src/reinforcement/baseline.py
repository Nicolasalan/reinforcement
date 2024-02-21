#!/usr/bin/env python3

import torch
import numpy as np

from agent import Agent
from environment import Env
from utils import Extension
from collections import deque

import rospy
import numpy as np
import os
import psutil
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
checkpoints_dir = os.path.join(script_dir, 'checkpoints')
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

def td3(n_episodes, print_every, max_t, score_solved, param, CONFIG_PATH, useful):
     """
     parameters
     ======
          n_episodes (int): maximum number of training episodes
          max_t(int): maximum number of timesteps per episode
     """

     state_dim = param["ENVIRONMENT_DIM"] + param["ROBOT_DIM"]
     action_dim = param["ACTION_DIM"]

     ## ====================== Training Loop ====================== ##

     if param["TYPE"] == 0:
               
          agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=0, CONFIG_PATH=CONFIG_PATH)

          agent.actor_local.load_state_dict(torch.load(param["TRAIN"] + "actor_model.pth", map_location=torch.device('cpu')))
          agent.critic_local.load_state_dict(torch.load(param["TRAIN"] + "critic_model.pth", map_location=torch.device('cpu')))

          env = Env(CONFIG_PATH)

          scores_window = deque()                                          # average scores of the most recent episodes                                                     
          scores = []                                                      # list of average scores of each episode                  

          for i_episode in range(n_episodes+1):                            # initialize score for each agent
               score = 0.0                
               done = False

               #agent.reset()                                               # reset environment    
               states = env.reset_env()                                    # get the current state of each agent
               
               for t in range(max_t):   
                    action = agent.action(states)                          # choose an action for each agent
                    actions = [(action[0] + 1) / 2, action[1]]             # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
                    next_states, rewards, done, _ = env.step_env(actions)  # send all actions to the environment
                    # save the experiment in the replay buffer, run the learning step at a defined interval
                    agent.step(states, actions, rewards, next_states, int(done), t, i_episode, scores)
                    states = next_states
                    score += rewards
                    if np.any(done) or t == max_t - 1:                                       # exit loop when episode ends
                         agent.learn(t)
                         break         
          
               scores_window.append(score)                                 # save average score for the episode
               scores.append(score)  
               mean_score = np.mean(scores_window)                          # save average score for the episode

               print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, mean_score, score), end="")

               if i_episode % 100 == 0:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))

               if i_episode % 300 == 0:
                    torch.save(agent.actor_local.state_dict(), os.path.join(checkpoints_dir, '{}_actor_checkpoint.pth'.format(i_episode)))
                    torch.save(agent.critic_local.state_dict(), os.path.join(checkpoints_dir, '{}_critic_checkpoint.pth'.format(i_episode)))

               if np.mean(scores_window) >= score_solved:
                    rospy.logwarn('Environment solved in ' + str(i_episode) + ' episodes!' + ' Average Score: ' + str(np.mean(scores_window)))
                    torch.save(agent.actor_local.state_dict(), os.path.join(checkpoints_dir, 'actor_checkpoint.pth'))
                    torch.save(agent.critic_local.state_dict(), os.path.join(checkpoints_dir, 'critic_checkpoint.pth'))
                    break

          return scores
     
          
if __name__ == '__main__':
     """Start training."""
     
     # folder to load config file        
     CONFIG_PATH = rospy.get_param('CONFIG_PATH')  
     useful = Extension(CONFIG_PATH)

     param = useful.load_config("config.yaml")

     n_episodes = param["N_EPISODES"]
     print_every = param["PRINT_EVERY"] 
     max_t = param["MAX_TIMESTEP"]
     score_solved = param["SCORE_SOLVED"]
     

     td3(n_episodes, print_every, max_t, score_solved, param, CONFIG_PATH, useful)