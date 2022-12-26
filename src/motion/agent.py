#!/usr/bin/env python3

import numpy as np
import random

from model import Actor, Critic
from noise import OUNoise
from replaybuffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
import os
import rospy

#import ray
#from ray import tune

# folder to load config file
CONFIG_PATH = "/ws/src/motion/config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        param = yaml.safe_load(file)

    return param

param = load_config("main_config.yaml")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
         
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        # TODO: raytune
        # Initialize Ray
        #ray.init()

        # Define the search space for hyperparameter optimization
        #config = {
        #    'lr': tune.uniform(0.001, 0.1),
        #    'l1': tune.uniform(32, 800),
        #    'l2': tune.uniform(32, 800),
        #    'gamma': tune.uniform(0.7, 0.99),
        #    'exploration': tune.uniform(0.1, 0.5),
        #    'tau': tune.uniform(0.001, 0.1),
        #    'batch_size': tune.quniform(16, 128, 16),
        #    'beta': tune.uniform(0.5, 0.9),
        #    'weight_decay': tune.uniform(0.0, 0.001),
        #    'momentum': tune.uniform(0.0, 0.9),
        #    'epsilon': tune.uniform(0.0, 1.0),
        #    'epsilon_decay': tune.uniform(0.99, 0.999)
        #}

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = param["EPSILON"]

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=param['LR_ACTOR'], weight_decay=0.0)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=param['LR_CRITIC'], weight_decay=0.0)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(int(param["BUFFER_SIZE"]), int(param["BATCH_SIZE"]), random_seed)
    
    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
 
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > param["BATCH_SIZE"] and 20.0 % float(param["LEARN_EVERY"]) == 0.0:
                 
            rospy.logwarn('Agent Learning               => Agent Learning ...')
            rospy.loginfo('Add Experience to Memory     => Experience: ' + str(len(self.memory)))

            for _ in range(param["LEARN_NUM"]):
                # Sample a batch of experiences from the replay buffer
                experiences, priorities, idxs = self.memory.sample()
                # Compute the loss and update the priorities
                loss = self.learn(experiences, float(param["GAMMA"]))
                loss_numpy = loss.detach().numpy()
                new_priorities = np.abs(loss_numpy) + 1e-5
                print(idxs, new_priorities)
                self.memory.update_priorities(idxs, new_priorities)
            
            rospy.loginfo('Get Target Q                 => Calculate Target Q ...')
        
    def action(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        state = torch.Tensor(state.reshape(1, -1)).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy().flatten()
        self.actor_local.train()

        if add_noise:
            action += self.epsilon * self.noise.sample()

        return np.clip(np.random.normal(action), -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """         
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)

        target_Q1, target_Q2 = self.critic_target(next_states, actions_next)
        # Select the minimal Q value from the 2 calculated values
        target_Q = torch.min(target_Q1, target_Q2)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + ((1 - dones) * gamma * target_Q).detach()
        # Compute critic loss
        Q1_expected, Q2_expected = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q1_expected, Q_targets) + F.mse_loss(Q2_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # normailize the gradient
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) 
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss, _ = self.critic_local(states, actions_pred)
        actor_loss = - actor_loss.mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, float(param["TAU"]))
        self.soft_update(self.actor_local, self.actor_target, float(param["TAU"])) 

        # ---------------------------- update noise ---------------------------- #
        self.epsilon -= float(param["EPSILON_DECAY"])
        self.noise.reset()  

        return critic_loss      

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
