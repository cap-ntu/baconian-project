from mobrl.core.basic import Basic
import abc
import numpy as np


class TerminalFunc(Basic):
    @abc.abstractmethod
    def __call__(self, state, action, new_state, **kwargs) -> bool:
        raise NotImplementedError

    def init(self):
        pass


class RandomTerminalFunc(Basic):
    """
    Debug and test use only
    """

    def __call__(self, state=None, action=None, new_state=None, **kwargs) -> bool:
        return np.random.random() > 0.5
