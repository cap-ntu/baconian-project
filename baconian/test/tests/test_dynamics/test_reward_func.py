from baconian.common.sampler.sample_data import TransitionData
from baconian.test.tests.set_up.setup import TestWithLogSet
import numpy as np
from baconian.algo.dynamics.terminal_func.terminal_func import *

x = 0


def func():
    return x


class TestRewardTerminalFunc(TestWithLogSet):

    def test_all_reward_func(self):
        pass

    def test_all_terminal_func(self):
        a = FixedEpisodeLengthTerminalFunc(max_step_length=10,
                                           step_count_fn=func)
        global x
        for i in range(11):
            if x == 10:
                self.assertTrue(a(state=None, action=None, new_state=None))
            else:
                self.assertFalse(a(state=None, action=None, new_state=None))
            x += 1
