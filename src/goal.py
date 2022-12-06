import torch
import torch.nn.functional as F
import torch.optim as optim
from agent import Agent
from environment import Env

state_dim = 16
action_dim = 2

env = Env()
agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=42)

agent.actor_local.load_state_dict(torch.load('actor_checkpoint.pth'))
agent.critic_local.load_state_dict(torch.load('critic_checkpoint.pth'))