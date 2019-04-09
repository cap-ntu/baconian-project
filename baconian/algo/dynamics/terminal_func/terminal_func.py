from baconian.core.core import Basic
import abc
import numpy as np


class TerminalFunc(Basic):
    allow_duplicate_name = True

    def __init__(self, name='terminal_func'):
        super().__init__(name=name)

    @abc.abstractmethod
    def __call__(self, state, action, new_state, **kwargs) -> bool:
        raise NotImplementedError

    def init(self):
        pass


class RandomTerminalFunc(TerminalFunc):
    """
    Debug and test use only
    """

    def __init__(self, name='random_terminal_func'):
        super().__init__(name)

    def __call__(self, state=None, action=None, new_state=None, **kwargs) -> bool:
        return np.random.random() > 0.5


class FixedEpisodeLengthTerminalFunc(Basic):

    def __init__(self, max_step_length: int, step_count_fn, status=None,
                 name: str = 'fixed_epsiode_length_terminal_func'):
        super().__init__(name, status)
        self.max_step_length = max_step_length
        self.step_count_fn = step_count_fn

    def __call__(self, state=None, action=None, new_state=None, **kwargs) -> bool:
        if self.step_count_fn() >= self.max_step_length:
            return True
        else:
            return False
