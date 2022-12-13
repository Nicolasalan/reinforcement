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
        print('Initializing agent')
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = param["EPSILON"]

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=float(param["LR_ACTOR"]))

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=float(param["LR_ACTOR"]), weight_decay=float(param["WEIGHT_DECAY"]))

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(int(param["BUFFER_SIZE"]), int(param["BATCH_SIZE"]), random_seed)
    
    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > param["BATCH_SIZE"] and timestep % float(param["LEARN_EVERY"]) == 0:
            for _ in range(param["LEARN_NUM"]):
                experiences = self.memory.sample()
                self.learn(experiences, float(param["GAMMA"]))

    def action(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        # Convert the state variable to a PyTorch tensor
        state = torch.Tensor(state)
        # Make sure the state tensor has the expected number of dimensions
        if state.ndim < 2:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]
        state_dim = state.shape[1]

        state = torch.Tensor(state.reshape(batch_size, state_dim)).to(device)

        self.actor_local.eval()
        with torch.no_grad():
            print('state: ', state)
            print('state shape: ', state.shape)
            print('state type: ', type(state))
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

        print('Learning')
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        target_Q1, target_Q2 = self.critic_target(next_states, actions_next)

        # Select the minimal Q value from the 2 calculated values
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = torch.stack(target_Q)

        # Compute Q targets for current states (y_i)
        Q_targets = float(rewards) + ((1 - dones) * gamma * target_Q).detach()
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # normailize the gradient
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) 
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
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
        
