from baconian.common.spaces.base import Space
# from baconian.common.random import Random
import numpy as np
from typeguard import typechecked
from baconian.core.parameters import Parameters


class ExplorationStrategy(object):
    def __init__(self):
        self.parameters = None

    def predict(self, **kwargs):
        raise NotImplementedError


class EpsilonGreedy(ExplorationStrategy):
    @typechecked
    def __init__(self, action_space: Space, init_random_prob: float):
        super(ExplorationStrategy, self).__init__()

        self.action_space = action_space
        self.init_random_prob = init_random_prob
        self.parameters = Parameters(parameters=dict(init_random_prob=self.init_random_prob),
                                     name='eps_greedy_params')

    def predict(self, **kwargs):
        if np.random.random() < self.parameters('init_random_prob'):
            return self.action_space.sample()
        else:
            algo = kwargs.pop('algo')
            return algo.predict(**kwargs)
