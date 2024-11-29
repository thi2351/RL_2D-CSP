import torch.nn as nn 
class A2CNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(A2CNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )



    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value
