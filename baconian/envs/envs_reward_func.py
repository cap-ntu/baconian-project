"""
Reward functions in most of some model-based methods are tricky problems,
since model itself require a reward function from outside.

Most of the codebase tune the definition of the reward function differed from original one
without explicit notifying which bring the difficult for users to tune the algorithms without the effect
brought by reward functions.

In Baconian, we try to clarify this part, make sure the user is well aware of the effect of such implementations.

This is a work in progress.

"""

from baconian.algo.dynamics.reward_func.reward_func import RewardFunc
import numpy as np


class PendulumRewardFunc(RewardFunc):

    def __init__(self, name='pendulum_reward_func'):
        super().__init__(name)
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05

    def __call__(self, state, action, new_state, **kwargs) -> float:
        th = state[0]
        thdot = state[1]
        u = np.clip(action, -self.max_torque, self.max_torque)[0]
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        return float(-costs)

    def init(self):
        super().init()


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


REWARD_FUNC_DICT = {
    'Pendulum-v0': PendulumRewardFunc
}
