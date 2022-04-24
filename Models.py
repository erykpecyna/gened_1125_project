import torch
import torch.nn as nn
import torch.nn.functional as F

class LunarQNetwork(nn.Module):
    def __init__(self):
        super(LunarQNetwork, self).__init__()
        self.fcn = nn.Sequential(
            nn.Linear(in_features = 8, out_features = 20),
            nn.ReLU(),
            nn.Linear(in_features = 20, out_features = 3)
        )
    
    def forward(self, x):
        out = self.fcn(x)
        return out