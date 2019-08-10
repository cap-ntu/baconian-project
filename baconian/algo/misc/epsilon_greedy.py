from baconian.common.spaces.base import Space
import numpy as np
from typeguard import typechecked
from baconian.core.parameters import Parameters
from baconian.common.schedules import Scheduler


class ExplorationStrategy(object):
    def __init__(self):
        self.parameters = None

    def predict(self, **kwargs):
        raise NotImplementedError


class EpsilonGreedy(ExplorationStrategy):
    @typechecked
    def __init__(self, action_space: Space, init_random_prob: float, prob_scheduler: Scheduler = None):
        super(ExplorationStrategy, self).__init__()

        self.action_space = action_space
        self.random_prob_func = lambda: init_random_prob
        if prob_scheduler:
            self.random_prob_func = prob_scheduler.value

        self.parameters = Parameters(parameters=dict(random_prob_func=self.random_prob_func),
                                     name='eps_greedy_params')

    def predict(self, **kwargs):
        if np.random.random() < self.parameters('random_prob_func')():
            return self.action_space.sample()
        else:
            algo = kwargs.pop('algo')
            return algo.predict(**kwargs)
