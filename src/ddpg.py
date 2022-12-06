#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from agent import Agent
from environment import Env

import rospy
import gym
import gym_gazebo
import numpy as np
import os
import yaml

# folder to load config file
CONFIG_PATH = "../config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        param = yaml.safe_load(file)

    return param


param = load_config("main_config.yaml")

state_dim = 16
action_dim = 2
action_linear_max = param["action_linear_max"]
action_angular_max = param["action_angular_max"]

print('State Dimensions: ' + str(state_dim))
print('Action Dimensions: ' + str(action_dim))
print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')


def ddpg():
     rospy.init_node('baseline')
     env = Env()
     past_action = np.array([0., 0.])
     """
     Parâmetros
     ======
         n_episodes (int): número máximo de episódios de treinamento
         max_t (int): número máximo de timesteps por episódio
     """
     agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=42)

     scores_window = deque()                                # pontuações médias dos episódios mais recentes
     scores = []                                            # lista de pontuações médias de cada episódio
                
     for i_episode in range(1, n_episodes+1):               # inicializar pontuação para cada agente
          agent.reset()                                     # redefinir ambiente
          states = env.reset()                              # obtém o estado atual de cada agente

          for t in range(max_t):
               actions = agent.action(states)                  # escolha uma ação para cada agente
               actions[0] = agent.action(states, 0.0, 1.0)                   # selecione uma ação
               actions[1] = agent.action(states,-0.5, 0.5)
               next_states, rewards, dones, arrive = env.step(actions, past_action) # envia todas as ações ao ambiente
               
               # salva a experiência no buffer de repetição, executa a etapa de aprendizado em um intervalo definido
               for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                    agent.step(state, action, reward, next_state, done, t)
                    
               states = next_states
               score += rewards
               past_action = actions
               if np.any(dones):                            # loop de saída quando o episódio termina
                    break              
               
          scores_window.append(score)                       # salvar pontuação média para o episódio
          scores.append(score)                              # salva pontuação média na janela
               
          print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_window)), end="") 
               
          if i_episode % print_every == 0:
               print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_window)))
          if np.mean(scores_window) >= score_solved:
               print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_window)))
               torch.save(agent.actor_local.state_dict(), 'actor_checkpoint.pth')
               torch.save(agent.critic_local.state_dict(), 'critic_checkpoint.pth')
               break

     return scores

if __name__ == '__main__':
     n_episodes = param["N_EPISODES"]
     print_every = param["PRINT_EVERY"] 
     max_t = param["MAX_T"]
     score_solved = param["SCORE_SOLVED"]

     scores = ddpg(n_episodes, print_every, max_t, score_solved)

     fig = plt.figure()
     ax = fig.add_subplot(111)
     plt.plot(np.arange(1, len(scores)+1), scores)
     plt.ylabel('Score')
     plt.xlabel('Episode')
     plt.show()