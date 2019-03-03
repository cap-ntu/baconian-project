from baconian.algo.rl.misc.exploration_strategy.base import ExplorationStrategy
from baconian.common.spaces.base import Space
from baconian.common.random import Random
from typeguard import typechecked
from baconian.core.parameters import Parameters
from baconian.common.util.schedules import Schedule, ConstantSchedule


class EpsilonGreedy(ExplorationStrategy):
    @typechecked
    def __init__(self, action_space: Space, init_random_prob: float,
                 random_state: Random = Random()):
        super(ExplorationStrategy, self).__init__()

        self.action_space = action_space
        self.init_random_prob = init_random_prob
        # todo decay for the prob
        self.random_state = random_state

        self.parameters = Parameters(parameters=dict(init_random_prob=self.init_random_prob),
                                    name='eps_greedy_params')

    def predict(self, **kwargs):
        if self.random_state.unwrapped().random() < self.parameters('init_random_prob'):
            return self.action_space.sample()
        else:
            algo = kwargs.pop('algo')
            return algo.predict(**kwargs)

