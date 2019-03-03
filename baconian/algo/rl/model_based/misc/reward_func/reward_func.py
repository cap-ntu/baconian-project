from baconian.core.core import Basic
import abc
import numpy as np


class RewardFunc(Basic):

    def __init__(self, name='reward_func'):
        super().__init__(name=name)

    @abc.abstractmethod
    def __call__(self, state, action, new_state, **kwargs) -> float:
        raise NotImplementedError

    def init(self):
        pass


class RandomRewardFunc(Basic):
    """
    Debug and test use only
    """

    def __init__(self, name='random_reward_func'):
        super().__init__(name)

    def __call__(self, state=None, action=None, new_state=None, **kwargs) -> float:
        return np.random.random()
