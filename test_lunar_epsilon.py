from tabnanny import check
import gym
import torch
from Models import LunarQNetwork_UCB

device = torch.device("cpu")

env = gym.make("LunarLander-v2")

model = LunarQNetwork_UCB()
checkpoint = torch.load("models/lunar_hybrid_1000.pt")
model.load_state_dict(checkpoint["model_state_dict"])

obs = env.reset()
total_reward = 0

while True:
    env.render()
    with torch.no_grad():
        obs = torch.tensor(obs).view(1,-1)
        vals = model.eval_actions(obs)
        a = int(vals)

        obs, reward, done, _ = env.step(a)
        total_reward += reward
        if done:
            break

print("Total Reward: {}".format(total_reward))
env.close()