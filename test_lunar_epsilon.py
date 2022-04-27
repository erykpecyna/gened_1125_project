from tabnanny import check
import gym
import torch
from Models import LunarQNetwork

device = torch.device("cpu")

env = gym.make("LunarLander-v2")

model = LunarQNetwork()
checkpoint = torch.load("models/lunar_epsilon_694.pt")
model.load_state_dict(checkpoint["model_state_dict"])

obs = env.reset()
total_reward = 0

while True:
    env.render()
    with torch.no_grad():
        obs = torch.tensor(obs).view(1,-1)
        vals = model(obs).argmax(dim=1)
        a = int(vals)

        obs, reward, done, _ = env.step(a)
        total_reward += reward
        if done:
            break

print("Total Reward: {}".format(total_reward))
env.close()