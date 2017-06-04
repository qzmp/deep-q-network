import numpy as np
from random import random, sample


class ExperienceBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        # if len(self.buffer) + len(experience) >= self.buffer_size:
        #     self.buffer[0:len(experience) + len(self.buffer) - self.buffer_size] = []
        # self.buffer.extend(experience)
        if len(self.buffer) > self.buffer_size:
            del self.buffer[0]
        self.buffer.append(experience)

    def sample(self, size):
        return np.array(sample(self.buffer, size))

    def __len__(self):
        return len(self.buffer)


class RandomActionChance:
    def __init__(self):
        self.startE = 1.0  # Starting chance of random action
        self.endE = 0.01  # Final chance of random action
        self.anneling_steps = 1000  # How many steps of training to reduce startE to endE.
        # Set the rate of random action decrease.
        self.e = self.startE
        self.stepDrop = (self.startE - self.endE) / self.anneling_steps

    def get_e(self):
        return self.e

    def step_down(self):
        if self.e > self.endE:
            self.e -= self.stepDrop

    def do_random_action(self):
        return random() < self.e
