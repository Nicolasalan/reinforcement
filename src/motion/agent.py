#!/usr/bin/env python3

from model import Actor, Critic
from noise import OUNoise
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


class Agent:
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

        self.tau = self.param["TAU"]
        self.epsilon = self.param["EPSILON"]
        self.clip_param = self.param["CLIP_PARAM"]
        self.max_action = self.param["MAX_ACTION"]
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

        # Actor Network (w/ Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters())

        # Critic Network (w/ Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_local.parameters())

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        self.av_Q = 0
        self.max_Q = -inf
        self.loss = 0
        self.iter = 0

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, random_seed)

    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        # print(self.memory.sample())

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size and int(done) > 0:
            rospy.logwarn("Agent Learning               => Agent Learning ...")
            rospy.loginfo(
                "Add Experience to Memory     => Experience: " + str(len(self.memory))
            )
            for steps in range(timestep + 1):
                # Sample a batch of experiences from the replay buffer
                # rospy.logwarn('Agent Learning               => Agent Learning ...')
                experiences = self.memory.sample()
                # Compute the loss and update the priorities
                self.learn(experiences, steps, self.policy_freq)

            self.useful.save_results("loss", self.loss / self.iter)
            self.useful.save_results("Av", self.av_Q / self.iter)
            self.useful.save_results("Max", self.max_Q / self.iter)

    def action(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor_local(state).cpu().data.numpy().flatten()

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, timestep, policy_freq):
        """Update policy and value parameters using given batch of experience tuples."""
        state, action, reward, next_state, done = experiences

        # Convert the batch to a torch tensor
        states = torch.Tensor(state).to(device)
        actions = torch.Tensor(action).to(device)
        rewards = torch.Tensor(reward).to(device)
        next_states = torch.Tensor(next_state).to(device)
        dones = torch.Tensor(done).to(device)

        # obtain the estimated action from next state by using the target actor network
        next_action = self.actor_target(next_states)

        # add noise to the estimated action
        noise = torch.Tensor(actions).data.normal_(0, self.policy_noise).to(device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_action = (next_action + noise).clamp(-1, 1)

        # ---------------------------- update critic ---------------------------- #
        # Calculate the Q values from the critic-target network for the next state-action pair
        target_Q1, target_Q2 = self.critic_target(next_states, next_action)

        # Select the minimal Q value from the 2 calculated values
        target_Q = torch.min(target_Q1, target_Q2)

        self.av_Q += torch.mean(target_Q)
        self.max_Q = max(self.max_Q, torch.max(target_Q))

        # normalization [-1, 1]
        rewards_norm = rewards / 100
        # print("rewards_norm", rewards_norm)

        # Calculate the final Q value from the target network parameters by using Bellman equation
        target_Q = (
            rewards_norm + ((1 - dones) * self.discount_factor * target_Q).detach()
        )

        # Get the Q values of the basis networks with the current parameters
        current_Q1, current_Q2 = self.critic_local(states, actions)

        # Calculate the loss between the current Q value and the target Q value
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        # print("critic_loss", critic_loss)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        if timestep % policy_freq == 0:
            # Compute actor loss
            actor_loss, _ = self.critic_local(states, self.actor_local(states))
            actor_loss = -actor_loss.mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # normailize the gradient
            self.actor_optimizer.step()

            # Update the critic target networks
            for param, target_param in zip(
                self.actor_local.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.critic_local.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        # ---------------------------- update noise ---------------------------- #
        self.epsilon -= float(self.epsilon_decay)
        self.loss += critic_loss
        self.iter += 1
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
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            local_param.data.copy_(
                tau * target_param.data + (1.0 - tau) * local_param.data
            )
