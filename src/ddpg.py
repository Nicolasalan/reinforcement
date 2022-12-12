#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from agent import Agent
from environment import Env

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

state_dim = param["state_dim"]
action_dim = param["action_dim"]
action_linear_max = param["action_linear_max"]
action_angular_max = param["action_angular_max"]

print('State Dimensions: ' + str(state_dim))
print('Action Dimensions: ' + str(action_dim))
print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')


def ddpg(n_episodes, print_every, max_t, score_solved):
     print('Starting DDPG')
     rospy.init_node('baseline-rl', anonymous=True)
     env = Env()
     """
     Parâmetros
     ======
         n_episodes (int): número máximo de episódios de treinamento
         max_t (int): número máximo de timesteps por episódio
     """
     agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=42)

     scores_window = []                                    # pontuações médias dos episódios mais recentes
     scores = 0.0                                          # lista de pontuações médias de cada episódio                                

     for i_episode in range(1, n_episodes+1):               # inicializar pontuação para cada agente
          score = 0.0
          print('episode: ' + str(i_episode))
          agent.reset()                                     # redefinir ambiente
          states = env.reset()                              # obtém o estado atual de cada agente

          for t in range(max_t):# escolha uma ação para cada agente
               print('timestep: ' + str(t))
               action = agent.action(states) 
               actions = [(action[0] + 1) / 2, action[1]]
               next_states, rewards, dones, _ = env.step(actions) # envia todas as ações ao ambiente
               
               # salva a experiência no buffer de repetição, executa a etapa de aprendizado em um intervalo definido
               agent.step(states, actions, rewards, next_states, dones, t)
               states = next_states
               score += rewards
               if np.any(dones):                           # loop de saída quando o episódio termina
                    break              
               
          scores_window.append(score)                   # salvar pontuação média para o episódio
          scores.append(score)                             # salva pontuação média na janela
               
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

     ddpg(n_episodes, print_every, max_t, score_solved)