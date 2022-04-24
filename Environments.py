
import gym

class LunarEnvironment:
    def __init__(self, config = {}):
        self.env = gym.make("LunarLander-v2")

    def start(self):
        self.rew_obs_term = (0.0, self.env.reset(), False)
        return self.rew_obs_term[1]
    
    def step(self, action):
        last_state = self.rew_obs_term[1]
        curr_state, reward, terminal, _ = self.env.step(action)
        self.rew_obs_term = reward, curr_state, terminal

        return self.rew_obs_term