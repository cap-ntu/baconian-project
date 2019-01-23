import gym
from src.core.basic import Basic
from typeguard import typechecked


class Env(gym.Env, Basic):
    """
    Abstract class for environment
    """
    key_list = []

    @typechecked
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = None
        self.observation_space = None
        self.step_count = None

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def init(self):
        raise NotImplementedError


if __name__ == '__main__':
    pass
