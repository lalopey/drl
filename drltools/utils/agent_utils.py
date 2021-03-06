import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = 'cpu'
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = self.convert_to_tensor(np.vstack([e.state for e in experiences if e is not None]))
        actions = self.convert_to_tensor(np.vstack([e.action for e in experiences if e is not None]))
        rewards = self.convert_to_tensor(np.vstack([e.reward for e in experiences if e is not None]))
        next_states = self.convert_to_tensor(np.vstack([e.next_state for e in experiences if e is not None]))
        dones = self.convert_to_tensor(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8))

        return states, actions, rewards, next_states, dones

    @staticmethod
    def convert_to_tensor(x):
        return torch.from_numpy(x).float().to(DEVICE)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
