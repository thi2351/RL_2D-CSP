import torch
import torch.nn as nn 
import gym
from Env import CuttingStockEnv

class A2C(nn.Module):
    def __init__(self, env:CuttingStockEnv, state_dim, hidden_dim):
        super(self, A2C).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.env = env  
        self.actor = self._get_actor_()
        self.critic = self._get_critic()

    def _get_actor(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dimm, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, len(self.env.rect)),
            nn.Softmax(dim = -1 )
        )
    
    def _get_critic(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLu(),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        critic_probs = self.critic(state)
        return action_probs, critic_probs