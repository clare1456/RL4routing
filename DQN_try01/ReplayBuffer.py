import collections
import random
import numpy as np

class ReplayBuffer():
    """经验回放池"""
    def __init__(self, capactity:int) -> None:
        self.memory = collections.deque(maxlen=capactity)

    def add(self, state, action, reward, next_state, done:bool):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size:int):
        transitions = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return np.array(states), actions, rewards, np.array(next_states), dones

