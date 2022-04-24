import random

from collections import deque

class Memory:
    def __init__(self, batch_size, memory_size):
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

    def append(self, state, action, terminal, reward, next_state):
        """
        Add the next experience to replay memory.

        Args:
            state: environment state (torch.Tensor)
            action: the action taken (not the index but the actual action)
            terminal: 1 if the next state is the terminal state
            reward: reward for this experience
            next_state: following state in the trajectory
        """
        self.memory.append((state, action, terminal, reward, next_state))

    def sample(self):
        """
        Return a sample of size batch_size for training
        """
        return random.sample(self.memory, self.batch_size)

    def get_memory(self):
        """
        Return the actual memory bufer
        """
        return self.memory

    def size(self):
        return len(self.memory)