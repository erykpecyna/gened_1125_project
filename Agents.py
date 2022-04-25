import torch
import torch.nn as nn
import numpy as np

from Models import LunarQNetwork, LunarQNetwork_UCB
from Memory import Memory
from utility import softmax, train_network, device

# Class design is inspired by https://github.com/sh2439/Reinforcement-Learning-Pytorch/blob/master/Lunar-Lander/LunarLander-Pytorch.ipynb
# but modified to suit my needs and updated libraries

# ---------------------------------------------------------------------------
# -                     LUNAR AGENTS                                        -
# ---------------------------------------------------------------------------

class LunarAgent:
    def __init__(self, config):
        self.model = LunarQNetwork().to(device)
        self.memory = Memory(config["batch_size"], config["memory_size"])
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = config["learning_rate"],
            betas = [0.99, 0.999],
            eps = 1e-4,
        )

        self.criterion = nn.MSELoss()

        self.config = config

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.episode_steps = 0
    
    def policy(self, state):
        with torch.no_grad():
            q = self.model(state) # First dim should be 1 (batch_size)
        p = softmax(q.data, self.config["tau"])
        p = np.array(p.cpu())
        p /= p.sum()
        a = np.random.choice(self.config["num_actions"], 1, p = p.squeeze())
        return a
    
    def start(self, state):
        self.sum_rewards = 0
        self.episode_steps = 0

        self.last_state = torch.tensor(state, device=device).unsqueeze(0)
        self.last_action = int(self.policy(self.last_state))

        return self.last_action
    
    def step(self, reward, state):
        """
        This is one step in the trajectory, only train if have enough samples in memory to do so
        """
        self.episode_steps += 1
        self.sum_rewards += reward

        state = torch.tensor(state, device=device).unsqueeze(0)
        action = self.policy(state)

        self.memory.append(self.last_state, self.last_action, 0, reward, state)

        # Action replay / train
        if self.memory.size() >= self.config["batch_size"]:
            target_model = LunarQNetwork().to(device)
            target_model.load_state_dict(self.model.state_dict())
            target_model.eval()

            # Do the actual replay
            for i in range(self.config["replay_iterations"]):
                experiences = self.memory.sample()

                # Training step for this replay episode
                train_network(
                    experiences,
                    self.model,
                    target_model,
                    self.optimizer,
                    self.criterion,
                    self.config["gamma"],
                    self.config["tau"]
                )
        
        self.last_state = state
        self.last_action = int(action)

        return self.last_action

    def end(self, reward):
        self.episode_steps += 1
        self.sum_rewards += reward

        state = torch.zeros_like(self.last_state)
        self.memory.append(self.last_state, self.last_action, 1, reward, state)

        # Action replay / train
        if self.memory.size() >= self.config["batch_size"]:
            target_model = LunarQNetwork().to(device)
            target_model.load_state_dict(self.model.state_dict())
            target_model.eval()

            # Do the actual replay
            for i in range(self.config["replay_iterations"]):
                experiences = self.memory.sample()

                # Training step for this replay episode
                train_network(
                    experiences,
                    self.model,
                    target_model,
                    self.optimizer,
                    self.criterion,
                    self.config["gamma"],
                    self.config["tau"]
                )
    
    def get_sum_rewards(self):
        return self.sum_rewards

class LunarAgent_UCB:
    def __init__(self, config):
        self.model = LunarQNetwork_UCB().to(device)
        self.memory = Memory(config["batch_size"], config["memory_size"])
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = config["learning_rate"],
            betas = [0.99, 0.999],
            eps = 1e-4,
        )

        self.criterion = nn.MSELoss()

        self.config = config

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.episode_steps = 0
    
    def policy(self, state):
        with torch.no_grad():
            a = self.model.sample_actions(state)
        return a
    
    def start(self, state):
        self.sum_rewards = 0
        self.episode_steps = 0

        self.last_state = torch.tensor(state, device=device).unsqueeze(0)
        self.last_action = int(self.policy(self.last_state))

        return self.last_action
    
    def step(self, reward, state):
        """
        This is one step in the trajectory, only train if have enough samples in memory to do so
        """
        self.episode_steps += 1
        self.sum_rewards += reward

        state = torch.tensor(state, device=device).unsqueeze(0)
        action = self.policy(state)

        self.memory.append(self.last_state, self.last_action, 0, reward, state)

        # Action replay / train
        if self.memory.size() >= self.config["batch_size"]:
            target_model = LunarQNetwork_UCB().to(device)
            target_model.load_state_dict(self.model.state_dict())
            target_model.eval()

            # Do the actual replay
            for i in range(self.config["replay_iterations"]):
                experiences = self.memory.sample()

                # Training step for this replay episode
                train_network(
                    experiences,
                    self.model,
                    target_model,
                    self.optimizer,
                    self.criterion,
                    self.config["gamma"],
                    self.config["tau"]
                )
        
        self.last_state = state
        self.last_action = int(action)

        return self.last_action

    def end(self, reward):
        self.episode_steps += 1
        self.sum_rewards += reward

        state = torch.zeros_like(self.last_state)
        self.memory.append(self.last_state, self.last_action, 1, reward, state)

        # Action replay / train
        if self.memory.size() >= self.config["batch_size"]:
            target_model = LunarQNetwork_UCB().to(device)
            target_model.load_state_dict(self.model.state_dict())
            target_model.eval()

            # Do the actual replay
            for i in range(self.config["replay_iterations"]):
                experiences = self.memory.sample()

                # Training step for this replay episode
                train_network(
                    experiences,
                    self.model,
                    target_model,
                    self.optimizer,
                    self.criterion,
                    self.config["gamma"],
                    self.config["tau"]
                )
    
    def get_sum_rewards(self):
        return self.sum_rewards

        