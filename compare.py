from tabnanny import check
from unittest import TestResult
import gym
import numpy as np
import torch
from tqdm import tqdm
from Models import LunarQNetwork_UCB, LunarQNetwork

env = gym.make("LunarLander-v2")

model_hybrid = LunarQNetwork_UCB()
model_epsilon = LunarQNetwork()
checkpoint_hybrid = torch.load("models/lunar_hybrid_582.pt")
checkpoint_epsilon = torch.load("models/lunar_epsilon_694.pt")

model_hybrid.load_state_dict(checkpoint_hybrid["model_state_dict"])
model_epsilon.load_state_dict(checkpoint_epsilon["model_state_dict"])

obs = env.reset()
total_reward = 0

TEST_RUNS = 500

hybrid_scores, epsilon_scores = [], []

for _ in tqdm(range(TEST_RUNS), desc="Hybrid Test", unit="ep"):
    obs = env.reset()
    total_reward = 0
    while True:
        with torch.no_grad():
            obs = torch.tensor(obs).view(1,-1)
            vals = model_hybrid.eval_actions(obs)
            a = int(vals)

            obs, reward, done, _ = env.step(a)
            total_reward += reward
            if done:
                break
    hybrid_scores.append(total_reward)

for _ in tqdm(range(TEST_RUNS), desc="Epsilon Test", unit="ep"):
    obs = env.reset()
    total_reward = 0
    while True:
        with torch.no_grad():
            obs = torch.tensor(obs).view(1,-1)
            vals = model_epsilon(obs)
            a = int(vals.argmax(dim=1))

            obs, reward, done, _ = env.step(a)
            total_reward += reward
            if done:
                break
    epsilon_scores.append(total_reward)

print("Average reward of Hybrid Method: {}".format(np.mean(hybrid_scores)))
print("Average reward of Epsilon Method: {}".format(np.mean(epsilon_scores)))
env.close()