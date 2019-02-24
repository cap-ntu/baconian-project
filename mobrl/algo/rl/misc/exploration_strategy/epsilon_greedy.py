from mobrl.algo.rl.misc.exploration_strategy.base import ExplorationStrategy
from mobrl.common.spaces.base import Space
from mobrl.common.random import Random
from typeguard import typechecked
from mobrl.core.parameters import Parameters


class EpsilonGreedy(ExplorationStrategy):
    @typechecked
    def __init__(self, action_space: Space, init_random_prob: float, decay_type: (int, None),
                 random_state: Random = Random()):
        self.action_space = action_space
        self.init_random_prob = init_random_prob
        self.decay_type = decay_type
        # todo decay for the prob
        self.random_state = random_state
        self.parameter = Parameters(parameters=dict(init_random_prob=self.init_random_prob,
                                                    decay_type=self.decay_type),
                                    name='eps_greedy_params')
        super(ExplorationStrategy, self).__init__()

    def predict(self, **kwargs):
        if self.random_state.unwrapped().random() < self.init_random_prob:
            return self.action_space.sample()
        else:
            algo = kwargs.pop('algo')
            return algo.predict(**kwargs)
