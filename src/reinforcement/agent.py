#!/usr/bin/env python3

from model import Actor, Critic
from replaybuffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim
import rospy
from numpy import inf
from utils import Extension
import numpy as np
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, CONFIG_PATH):
        """Initialize an Agent object.
         
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.useful = Extension(CONFIG_PATH)
        # Function to load yaml configuration file
        self.param = self.useful.load_config("config.yaml")
        self.seed = np.random.seed(random_seed)

        self.action_size = action_size
        self.state_size = state_size

        self.noise=0.2
        self.noise_std=0.1
        self.noise_clip=0.5
        
        self.tau = self.param["TAU"]
        self.epsilon = self.param["EPSILON"]
        self.clip_param= self.param["CLIP_PARAM"]
        self.max_action = 1
        self.min_action = -1
        self.discount_factor = self.param["DISCOUNT"]
        self.batch_size = self.param["BATCH_SIZE"]
        self.lr_actor = self.param["LR_ACTOR"]
        self.lr_critic = self.param["LR_CRITIC"]
        self.weight_decay = self.param["WEIGHT_DECAY"]
        self.buffer_size = self.param["BUFFER_SIZE"]
        self.policy_freq = self.param["POLICY_FREQ"]
        self.epsilon_decay = self.param["EPSILON_DECAY"]
        self.policy_noise = self.param["POLICY_NOISE"]
        self.noise_clip = self.param["NOISE_CLIP"]
        self.gamma = 0.99

        # Actor Network (w/ Network)
        self.actor_local = Actor().to(device)
        self.actor_target = Actor().to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.param['LR_ACTOR'])

        # Critic Network (w/ Network)
        self.critic_local = Critic().to(device)
        self.critic_target = Critic().to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.param['LR_CRITIC'])

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, random_seed)
    
    def step(self, state, action, reward, next_state, done, timestep, i_episode, score):
        """Save experience in replay memory"""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
    def action(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        action = self.actor_local(torch.as_tensor(state, dtype=torch.float32).view(1, -1)).cpu().data.numpy().flatten()
        return action        
        # state = torch.from_numpy(state).float().to(device)
        # self.actor_local.eval()
        # with torch.no_grad():
        #     action = self.actor_local(state).cpu().data.numpy()
        # if add_noise:
        #     # Generate a random noise
        #     noise = np.random.normal(0, self.noise_std, size=self.action_size)
        #     # Add noise to the action for exploration
        #     action = (action + noise).clip(self.min_action, self.max_action)
        # self.actor_local.train()
        # return action

    def learn(self, n_iteraion, ):

        if len(self.memory) > self.batch_size:
            for i in range(n_iteraion):
                state, action, reward, next_state, done = self.memory.sample()

                action_ = action.cpu().numpy()

                # ---------------------------- update critic ---------------------------- #
                # Get predicted next-state actions and Q values from target models
                actions_next = self.actor_target(next_state)

                # Generate a random noise
                noise = torch.FloatTensor(action_).data.normal_(0, self.noise).to(device)
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                actions_next = (actions_next + noise).clamp(-1.0, 1) # mudar aqui depois

                Q1_targets_next, Q2_targets_next = self.critic_target(next_state, actions_next)

                Q_targets_next = torch.min(Q1_targets_next, Q2_targets_next)
                # Compute Q targets for current states (y_i)
                Q_targets = reward + (self.gamma * Q_targets_next * (1 - done)).detach()
                # Compute critic loss
                Q1_expected, Q2_expected = self.critic_local(state, action)
                critic_loss = F.mse_loss(Q1_expected, Q_targets) + F.mse_loss(Q2_expected, Q_targets)

                # Minimize the loss
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                if i % self.policy_freq == 0:
                    # ---------------------------- update actor ---------------------------- #
                    # Compute actor loss
                    actor_loss, _ = self.critic_local(state, self.actor_local(state))
                    actor_loss = -actor_loss.mean()
                    # Minimize the loss
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # ----------------------- update target networks ----------------------- #
                    self.soft_update(self.critic_local, self.critic_target, self.tau)
                    self.soft_update(self.actor_local, self.actor_target, self.tau)

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
            local_param.data.copy_(tau*target_param.data + (1.0-tau)*local_param.data)
