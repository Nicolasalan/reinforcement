#!/usr/bin/env python3

from model import Actor, Critic
from noise import OUNoise
from replaybuffer import ReplayBuffer

from collections import OrderedDict, deque, namedtuple
import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy import inf
from utils import Extension
import numpy as np
from lightning.pytorch import cli_lightning_logo, LightningModule, seed_everything, Trainer

Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)
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

        self.av_Q = 0.0
        self.max_Q = -inf
        self.loss = 0.0
        self.iter = 0.0

        # Replay memory
        self.memory = ReplayBuffer(capacity=self.buffer_size)

    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size and int(done) > 0:

            if self.memory.tree.n_entries >= self.batch_size:
                for steps in range(timestep + 1):
                    # Sample a batch of experiences from the replay buffer
                    experiences, idxs, weights = self.sample()
                    
                    # Compute the loss and update the priorities
                    self.learn(experiences, steps, idxs, weights, self.policy_freq)

                self.useful.save_results("loss", self.loss / self.iter)
                self.useful.save_results("Av", self.av_Q / self.iter)
                self.useful.save_results("Max", self.max_Q / self.iter)

    def action(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        state = torch.Tensor(state.reshape(1, -1)).to(device)
        action = self.actor_local(state).cpu().data.numpy().flatten()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -self.max_action, self.max_action)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, timestep, idxs, weights, policy_freq):
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
        state, action, reward, next_state, done = experiences
        #experiences = koila.lazy((state, action, reward, next_state, done), batch=0)

        # Convert the batch to a torch tensor
        states = torch.Tensor(state).to(device)
        actions = torch.Tensor(action).to(device)
        rewards = torch.Tensor(reward).to(device)
        next_states = torch.Tensor(next_state).to(device)
        dones = torch.Tensor(done).to(device)

        # obtain the estimated action from next state by using the target actor network
        next_action = self.actor_target(next_states)

        # ---------------------------- update critic ---------------------------- #
        # Calculate the Q values from the critic-target network for the next state-action pair
        target_Q1, target_Q2 = self.critic_target(next_states, next_action)

        # Select the minimal Q value from the 2 calculated values
        target_Q = torch.min(target_Q1, target_Q2)

        self.av_Q += torch.mean(target_Q)
        self.max_Q = max(self.max_Q, torch.max(target_Q))

        # normalization [-1, 1]
        rewards_norm = rewards / 100

        # Calculate the final Q value from the target network parameters by using Bellman equation
        target_Q = (
            rewards_norm + ((1 - dones) * self.discount_factor * target_Q).detach()
        )

        # Get the Q values of the basis networks with the current parameters
        current_Q1, current_Q2 = self.critic_local(states, actions)

        Q_expected = torch.min(current_Q1, current_Q2)

        # Calculate the loss between the current Q value and the target Q value
        critic_loss = torch.from_numpy(weights).float().to(device) * F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Introducing gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) 

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
            self.soft_update(self.actor_local, self.actor_target, self.tau)

            self.soft_update(self.critic_local, self.critic_target, self.tau)
            
        # ---------------------------- update noise ---------------------------- #
        self.epsilon -= float(self.epsilon_decay)
        self.loss += critic_loss
        self.iter += 1
        self.noise.reset()

        # ---------------------------- update Buffer --------------------------- #
        # Calculate errors used in prioritized replay buffer
        errors = (Q_expected-target_Q).squeeze().cpu().data.numpy()
        
        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

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

    
class Buffer:

    def append(self, state, action, reward, next_state, done, gamma) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """

        next_state = torch.from_numpy(next_state, dtype=np.float32).to(device)
        action     = torch.from_numpy(action, dtype=np.float32).to(device)
        state      = torch.from_numpy(state, dtype=np.float32).to(device)
        reward     = torch.from_numpy(reward, dtype=np.float32).to(device)
        done       = torch.from_numpy(done, dtype=np.bool).to(device)

        # reshape the tensors
        reward = torch.unsqueeze(reward, 1)
        done = torch.unsqueeze(done, 1)

        self.actor_target.eval(), self.critic_target.eval(), self.critic_local.eval()

        with torch.no_grad():
            action_next = self.actor_target(next_state)
            Q_target_next = self.critic_target(next_state, action)
            Q_target = reward + (gamma * Q_target_next * (1 -done))
            Q_expected = self.critic_local(state, action)

        self.actor_local.train(), self.critic_target.train(), self.critic_local.train()

        # Error used in prioritized replay buffer
        error = (Q_expected - Q_target).squeeze().cpu().data.numpy()

        #Adding experiences to prioritized replay buffer
        for i in np.arange(len(reward)):
            self.memory.add(error[i], (state[i], action[i], reward[i], next_state[i], done[i]))
    

    def sample(self, batch_size: int):
        """Randomly sample a batch of experiences from memory."""
        experiences, idxs, weights = self.memory.sample(batch_size)

        states = np.vstack([e[0] for e in experiences])
        states = torch.from_numpy(states).float().to(device)
        
        actions = np.vstack([e[1] for e in experiences])
        actions = torch.from_numpy(actions).float().to(device)
        
        rewards = np.vstack([e[2] for e in experiences])
        rewards = torch.from_numpy(rewards).float().to(device)
        
        next_states = np.vstack([e[3] for e in experiences])
        next_states = torch.from_numpy(next_states).float().to(device)
        
        dones = np.vstack([e[4] for e in experiences]).astype(np.uint8)
        dones = torch.from_numpy(dones).float().to(device)

        return (states, 
                actions, 
                rewards, 
                next_states, 
                dones
                ), idxs, weights    

class TD3(LightningModule):

    def __init__(
        self,
        env: str,
        replay_size: int = 200,
        warm_start_steps: int = 200,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_last_frame: int = 200,
        sync_rate: int = 10,
        lr: float = 1e-2,
        episode_length: int = 50,
        batch_size: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.replay_size = replay_size
        self.warm_start_steps = warm_start_steps
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_last_frame = eps_last_frame
        self.sync_rate = sync_rate
        self.lr = lr
        self.episode_length = episode_length
        self.batch_size = batch_size

        self.env = gym.make(env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.
        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes in a state `x` through the network and gets the `q_values` of each action as an output.
        Args:
            x: environment state
        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch received.
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(self.eps_end, self.eps_start - (self.global_step + 1) / self.eps_last_frame)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "steps": torch.tensor(self.global_step).to(device),
        }

        return OrderedDict({"loss": loss, "log": log, "progress_bar": log})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=None)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

def main(args) -> None:
    model = TD3Lightning(**vars(args))
    trainer = Trainer(accelerator="cpu", devices=1, val_check_interval=100)
    trainer.fit(model)

