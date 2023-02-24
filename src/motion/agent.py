#!/usr/bin/env python3

import numpy as np
import random

from model import Actor, Critic
from noise import OUNoise
from replaybuffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F
import torch.optim as optim
import rospy
from numpy import inf
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
        self.writer = SummaryWriter()

        self.av_Q = 0
        self.max_Q = -inf

        # Replay memory
        self.memory = ReplayBuffer(int(self.param["BUFFER_SIZE"]), int(self.param["BATCH_SIZE"]), random_seed)
    
    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
 
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.param["BATCH_SIZE"] and timestep % float(self.param["LEARN_EVERY"]) == 0.0:
                 
            #rospy.logwarn('Agent Learning               => Agent Learning ...')
            rospy.loginfo('Add Experience to Memory     => Experience: ' + str(len(self.memory)))
            for _ in range(self.param["LEARN_NUM"]):
                # Sample a batch of experiences from the replay buffer
                experiences = self.memory.sample()
                # Compute the loss and update the priorities
                loss = self.learn(experiences, timestep, self.param["POLICY_FREQ"])
                loss_numpy = loss.detach().numpy()
            
            rospy.loginfo('Calculate Loss               => Loss: ' + str(loss_numpy))
        
    def action(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        state = torch.Tensor(state.reshape(1, -1)).to(device)
        #self.actor_local.eval()
        #with torch.no_grad():
        action = self.actor_local(state).cpu().data.numpy().flatten()
        #self.actor_local.train()

        # ! state
        # ! tensor([[10.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000,
        # ! 10.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000,
        # ! 10.0000, 10.0000, 10.0000, 10.0000,  3.4557,  2.2288,  0.0000,  0.0000]])

        if add_noise:
            action += self.epsilon * self.noise.sample()

        return np.clip(np.random.normal(action), -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, timestep, policy_freq):
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Calculate the Q values from the critic-target network for the next state-action pair
        target_Q1, target_Q2 = self.critic_target(next_states, actions)

        # ! Next Action tensor([[-0.2305,  0.5880], [-0.2348,  0.5885], [-0.2202,  0.5964]], grad_fn=<TanhBackward>)
        # ! Next Action noise tensor([[-0.3595,  0.7761], [-0.4474,  0.4941], [-0.1140,  0.5544]], grad_fn=<ClampBackward1>)

        # ! target_Q1 tensor([[0.5569], [0.5400], [0.6051]], grad_fn=<AddmmBackward>)
        # ! target_Q2 tensor([[0.3943], [0.3690], [0.4310]], grad_fn=<AddmmBackward>)

        # Select the minimal Q value from the 2 calculated values
        target_Q = torch.min(target_Q1, target_Q2)

        self.av_Q += torch.mean(target_Q)
        self.max_Q = max(self.max_Q, torch.max(target_Q))

        # Calculate the final Q value from the target network parameters by using Bellman equation
        target_Q = rewards + ((1 - dones) * self.discount_factor * target_Q).detach() # ! tensor([[ 0.4580], [ 0.6587], [-0.0690]])

        # Get the Q values of the basis networks with the current parameters
        current_Q1, current_Q2 = self.critic_local(states, actions)
        # ! loss tensor(0.5476, grad_fn=<AddBackward0>)

        # Calculate the loss between the current Q value and the target Q value
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        if timestep % policy_freq == 0:
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss, _ = self.critic_local(states, actions_pred)
            actor_loss = -actor_loss.mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # normailize the gradient
            self.actor_optimizer.step()

            # Update the critic target networks
            self.soft_update(self.critic_local, self.critic_target, self.tau)

            # Update the critic target networks
            self.soft_update(self.actor_local, self.actor_target, self.tau)

            # Write new values for tensorboard
            self.writer.add_scalar("Loss", critic_loss / timestep)
            self.writer.add_scalar("Av. Q", self.av_Q / timestep)
            self.writer.add_scalar("Max. Q", self.max_Q, timestep)
            self.writer.add_scalar("Rewards", rewards.mean().reshape(1), timestep)

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