import torch
import torch.nn as nn
import torch.nn.functional as F

class LunarQNetwork(nn.Module):
    def __init__(self):
        super(LunarQNetwork, self).__init__()
        self.fcn = nn.Sequential(
            nn.Linear(in_features = 8, out_features = 20),
            nn.ReLU(),
            nn.Linear(in_features = 20, out_features = 4)
        )
    
    def forward(self, x):
        out = self.fcn(x)
        return out

class LunarQNetwork_UCB(nn.Module):
    def __init__(self):
        super(LunarQNetwork_UCB, self).__init__()
        self.fcn = nn.Sequential(
            nn.Linear(in_features = 8, out_features = 20),
            nn.ReLU(),
        )
        self.mu = nn.Linear(in_features = 20, out_features = 4)
        self.sigma = nn.Linear(in_features = 20, out_features = 4)
    
    def forward(self, x):
        x = self.fcn(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return torch.randn(mu.shape, device = mu.device) * sigma + mu

    def sample_actions(self, x):
        x = self.fcn(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        ucb = mu + sigma.abs()
        return ucb.argmax(dim=1)

    def eval_actions(self, x):
        x = self.fcn(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        ucb = mu
        return ucb.argmax(dim=1)