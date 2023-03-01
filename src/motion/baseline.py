#!/usr/bin/env python3

import torch
import numpy as np

from agent import Agent
from environment import Env
from continuous import ContinuousEnv
from utils import Extension
import psutil

import rospy
import numpy as np
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
checkpoints_dir = os.path.join(script_dir, 'checkpoints')
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

def ddpg(n_episodes, print_every, max_t, score_solved, param, CONFIG_PATH, count):
     """
     parameters
     ======
          n_episodes (int): maximum number of training episodes
          max_t(int): maximum number of timesteps per episode
     """

     state_dim = param["ENVIRONMENT_DIM"] + param["ROBOT_DIM"]
     action_dim = param["ACTION_DIM"]

     count_rand_actions = 0
     random_action = []

     ## ====================== Training Loop ====================== ##

     if param["TYPE"] == 0:
          print('Training DDPG in simulation')
               
          agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=42, CONFIG_PATH=CONFIG_PATH)
          env = Env(CONFIG_PATH)

          scores_window = []                                               # average scores of the most recent episodes
          scores = []                                                      # list of average scores of each episode                     

          for i_episode in range(n_episodes+1):                            # initialize score for each agent
               #rospy.loginfo('Episode: ' + str(i_episode))
               score = 0.0                
               done = False

               agent.reset()                                               # reset environment    
               states = env.reset_env()                                    # get the current state of each agent

               
               for t in range(max_t):   
                         
                    action = agent.action(states)                          # choose an action for each agent
                    
                    actions = [(action[0] + 1) / 2, action[1]]             # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
                    #action_random = useful.random_near_obstacle(states, count_rand_actions, random_action, param["random_near_obstacle"])

                    next_states, rewards, done, _ = env.step_env(actions)  # send all actions to the environment

                    # save the experiment in the replay buffer, run the learning step at a defined interval
                    agent.step(states, actions, rewards, next_states, done, t)

                    states = next_states
                    score += rewards
                    if np.any(done):                                       # exit loop when episode ends
                         break              
                    
                    scores_window.append(score)                            # save average score for the episode
                    scores.append(score)                                   # save average score in the window  
                         
               cpu_usage = psutil.cpu_percent()
               rospy.loginfo('CPU and Memory               => usage: ' + str(cpu_usage) + '%, ' + str(psutil.virtual_memory().percent) + '%')
               
               if i_episode % print_every == 0:
                    rospy.loginfo('# ====== Episode: ' + str(i_episode) + ' Average Score: ' + str(np.mean(scores_window)) + ' ====== #')
               
               if i_episode % 100 == 0:
                    torch.save(agent.actor_local.state_dict(), os.path.join(checkpoints_dir, '{}_actor_checkpoint.pth'.format(i_episode)))
                    torch.save(agent.critic_local.state_dict(), os.path.join(checkpoints_dir, '{}_critic_checkpoint.pth'.format(i_episode)))

               if np.mean(scores_window) >= score_solved:
                    rospy.logwarn('Environment solved in ' + str(i_episode) + ' episodes!' + ' Average Score: ' + str(np.mean(scores_window)))
                    torch.save(agent.actor_local.state_dict(), os.path.join(checkpoints_dir, 'actor_checkpoint.pth'))
                    torch.save(agent.critic_local.state_dict(), os.path.join(checkpoints_dir, 'critic_checkpoint.pth'))
                    break

          return scores

     ## ====================== Continuous Training ====================== ##

     if param["TYPE"] == 1:
          print('Training continuous DDPG in simulation')

          agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=42, CONFIG_PATH=CONFIG_PATH)
          env = Env(CONFIG_PATH)

          agent.actor_local.load_state_dict(torch.load(param["MODEL"] + 'actor_model.pth'))
          agent.critic_local.load_state_dict(torch.load(param["MODEL"] + 'critic_model.pth'))

          scores_window = []                                               # average scores of the most recent episodes
          scores = []                                                      # list of average scores of each episode                     

          for i_episode in range(n_episodes+1):                            # initialize score for each agent
               #rospy.loginfo('Episode: ' + str(i_episode))
               score = 0.0                
               done = False

               agent.reset()                                               # reset environment    
               states = env.reset_env()                                    # get the current state of each agent

               for t in range(max_t):   
                         
                    action = agent.action(states)                          # choose an action for each agent
                    actions = [(action[0] + 1) / 2, action[1]]             # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity

                    next_states, rewards, done, _ = env.step_env(actions)  # send all actions to the environment
                    #action_random = useful.random_near_obstacle(states, count_rand_actions, random_action, param["random_near_obstacle"])

                    # save the experiment in the replay buffer, run the learning step at a defined interval
                    agent.step(states, actions, rewards, next_states, done, t)

                    states = next_states
                    score += rewards
                    if np.any(done):                                       # exit loop when episode ends
                         break              
                    
               scores_window.append(score)                            # save average score for the episode
               scores.append(score)                                   # save average score in the window  
                         
               cpu_usage = psutil.cpu_percent()
               rospy.loginfo('CPU and Memory               => usage: ' + str(cpu_usage) + '%, ' + str(psutil.virtual_memory().percent) + '%')
               
               if i_episode % print_every == 0:
                    rospy.loginfo('# ====== Episode: ' + str(i_episode) + ' Average Score: ' + str(np.mean(scores_window)) + ' ====== #')
               
               if i_episode % 100 == 0:
                    torch.save(agent.actor_local.state_dict(), os.path.join(checkpoints_dir, '{}_actor_checkpoint.pth'.format(i_episode)))
                    torch.save(agent.critic_local.state_dict(), os.path.join(checkpoints_dir, '{}_critic_checkpoint.pth'.format(i_episode)))

               if np.mean(scores_window) >= score_solved:
                    rospy.logwarn('Environment solved in ' + str(i_episode) + ' episodes!' + ' Average Score: ' + str(np.mean(scores_window)))
                    torch.save(agent.actor_local.state_dict(), os.path.join(checkpoints_dir, 'actor_checkpoint.pth'))
                    torch.save(agent.critic_local.state_dict(), os.path.join(checkpoints_dir, 'critic_checkpoint.pth'))
                    break

          return scores
     
     ## ====================== Test Environment ====================== ##
     
     if param["TYPE"] == 2:
          print('Testing DDPG in real environment')
     
          agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=42, CONFIG_PATH=CONFIG_PATH)
          env = ContinuousEnv(CONFIG_PATH)

          agent.actor_local.load_state_dict(torch.load(param["MODEL"] + 'actor_model.pth'))
          agent.critic_local.load_state_dict(torch.load(param["MODEL"] + 'critic_model.pth'))

          scores_window = []                                               # average scores of the most recent episodes
          scores = []                                                      # list of average scores of each episode

          num_resets = 0   

          while num_resets < count:                                        # initialize score for each agent
               done = False       

               agent.reset()                                               # reset environment    
               states = env.reset_env()                                    # get the current state of each agent

               max_t = 10000
               for t in range(max_t):   
                         
                    action = agent.action(states)                          # choose an action for each agent
                    actions = [(action[0] + 1) / 2, action[1]]             # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity

                    next_states, rewards, done, _ = env.step_env(actions)  # send all actions to the environment

                    # save the experiment in the replay buffer, run the learning step at a defined interval
                    agent.step(states, actions, rewards, next_states, done, t)

                    states = next_states
                    if np.any(done):                                       # exit loop when episode ends
                         break              

               num_resets += 1 
                         
               cpu_usage = psutil.cpu_percent()
               rospy.logwarn('CPU and Memory               => usage: ' + str(cpu_usage) + '%, ' + str(psutil.virtual_memory().percent) + '%')

          return scores
          
if __name__ == '__main__':
     """Start training."""
     
     # folder to load config file        
     CONFIG_PATH = rospy.get_param('CONFIG_PATH')  
     useful = Extension(CONFIG_PATH)

     param = useful.load_config("config.yaml")
     count = len(useful.path_target(param["CONFIG_PATH"] + "poses.yaml"))

     n_episodes = param["N_EPISODES"]
     print_every = param["PRINT_EVERY"] 
     max_t = param["MAX_T"]
     score_solved = param["SCORE_SOLVED"]

     ddpg(n_episodes, print_every, max_t, score_solved, param, CONFIG_PATH, count)