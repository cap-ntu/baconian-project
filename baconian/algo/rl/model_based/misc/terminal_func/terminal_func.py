from baconian.core.core import Basic
import abc
import numpy as np


class TerminalFunc(Basic):

    def __init__(self, name='terminal_func'):
        super().__init__(name=name)

    @abc.abstractmethod
    def __call__(self, state, action, new_state, **kwargs) -> bool:
        raise NotImplementedError

    def init(self):
        pass


class RandomTerminalFunc(Basic):
    """
    Debug and test use only
    """
    def __init__(self, name='random_terminal_func'):
        super().__init__(name)

    def __call__(self, state=None, action=None, new_state=None, **kwargs) -> bool:
        return np.random.random() > 0.5
