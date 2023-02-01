#!/usr/bin/env python3

import numpy as np
import random

from model import Actor, Critic
from noise import OUNoise
from replaybuffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim
import rospy
from utils import Extension

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        useful = Extension(CONFIG_PATH)
        # Function to load yaml configuration file
        self.param = useful.load_config("config.yaml")

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.tau = self.param["TAU"]
        self.epsilon = self.param["EPSILON"]
        self.clip_param= self.param["CLIP_PARAM"]
        self.max_action = self.param["MAX_ACTION"]
        self.discount_factor = self.param["DISCOUNT"]

        # Actor Network (w/ Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.param['LR_ACTOR'])

        # Critic Network (w/ Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.param['LR_CRITIC'], weight_decay=self.param['WEIGHT_DECAY'])

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(int(self.param["BUFFER_SIZE"]), int(self.param["BATCH_SIZE"]), random_seed)
    
    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
 
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.param["BATCH_SIZE"] and timestep % float(self.param["LEARN_EVERY"]) == 0.0:
                 
            rospy.logwarn('Agent Learning               => Agent Learning ...')
            rospy.loginfo('Add Experience to Memory     => Experience: ' + str(len(self.memory)))
            for _ in range(self.param["LEARN_NUM"]):
                # Sample a batch of experiences from the replay buffer
                experiences = self.memory.sample()
                # Compute the loss and update the priorities
                loss = self.learn(experiences, timestep, self.param["POLICY_NOISE"], self.param["POLICY_FREQ"])
                loss_numpy = loss.detach().numpy()
            
            rospy.loginfo('Calculate Loss               => Loss: ' + str(loss_numpy))
        
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

    def learn(self, experiences, timestep, policy_noise, policy_freq):
        states, actions, rewards, next_states, n_step_returns, dones = experiences

        # ---------------------------- update critic ---------------------------- #

        # Compute Q targets for current states (y_i) using the n-step returns
        Q_targets = n_step_returns 
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
        if timestep % policy_freq == 0:
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss, _ = self.critic_local(states, actions_pred)
            actor_loss = - actor_loss.mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Use the newly updated actor to generate target actions and add noise
            noise = (torch.randn_like(actions) * policy_noise).clamp(-self.clip_param, self.clip_param)
            target_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

            # Compute the minimum of the two Q values for the target actions
            target_Q1, target_Q2 = self.critic_target(next_states, target_actions)
            target_Q = torch.min(target_Q1, target_Q2)

            # Compute the target n-step return
            n_step_return = (self.discount_factor ** timestep) * target_Q * (1 - dones)
            n_step_return += rewards

            # Update the critic target networks
            self.soft_update(self.critic_local, self.critic_target, self.tau)

        # ---------------------------- update noise ---------------------------- #
        self.epsilon -= float(self.param["EPSILON_DECAY"])
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