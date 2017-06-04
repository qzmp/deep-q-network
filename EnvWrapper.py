import gym
import numpy as np


class PongWrapper:
    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.state_shape = self.reset().shape
        self.input_num = self.env.action_space.n

    @staticmethod
    def _process_state(state):
        return np.expand_dims(state.mean(2), 2) / 255

    def reset(self):
        state = self.env.reset()
        return self._process_state(state)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        state = self._process_state(state)
        return state, reward, done

    def render(self):
        self.env.render()

    @property
    def action_space(self):
        return self.env.action_space
