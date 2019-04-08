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


class RandomTerminalFunc(TerminalFunc):
    """
    Debug and test use only
    """
    def __init__(self, name='random_terminal_func'):
        super().__init__(name)

    def __call__(self, state=None, action=None, new_state=None, **kwargs) -> bool:
        return np.random.random() > 0.5


class FixedEpisodeLengthTerminalFunc(Basic):

    def __init__(self, name: str, max_step_length: int, step_count_fn, status=None):
        super().__init__(name, status)
        self.max_step_length = max_step_length
        self.step_count_fn = step_count_fn
        self._last_reset_point = -1

    def __call__(self, state=None, action=None, new_state=None, **kwargs) -> bool:
        if self.step_count_fn() - self._last_reset_point >= self.max_step_length:
            self._last_reset_point = self.step_count_fn()
            return True
        else:
            return False
