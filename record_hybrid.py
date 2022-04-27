from email.mime import base
from tabnanny import check
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import torch
from Models import LunarQNetwork_UCB

device = torch.device("cpu")

env = gym.make("LunarLander-v2")
vr = VideoRecorder(env, base_path="videos/hybrid")

model = LunarQNetwork_UCB()
checkpoint = torch.load("models/lunar_hybrid_582.pt")
model.load_state_dict(checkpoint["model_state_dict"])

obs = env.reset()
total_reward = 0

while True:
    vr.capture_frame()
    with torch.no_grad():
        obs = torch.tensor(obs).view(1,-1)
        vals = model.eval_actions(obs)
        a = int(vals)

        obs, reward, done, _ = env.step(a)
        total_reward += reward
        if done:
            break

vr.close()
print("Total Reward: {}".format(total_reward))
env.close()