import gym
from src.core.basic import Basic
from gym.core import Space
import numpy as np
from src.core.config import config
from typeguard import typechecked


class Env(gym.Env, Basic):
    """
    Abstract class for environment
    """
    key_list = []

    @typechecked
    def __init__(self, config: config):
        super(Env, self).__init__(config=config)
        self.action_space = Space()
        self.observation_space = Space()
        self.cost_fn = None
        self.step_count = 0

    def step(self, action):
        return None

    def reset(self):
        return None

    def init(self):
        print("%s init finished" % type(self).__name__)
        return True


if __name__ == '__main__':
    pass
