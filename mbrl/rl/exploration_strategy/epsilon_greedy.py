from mbrl.rl.exploration_strategy.base import ExplorationStrategy
from gym.core import Space
from mbrl.common.random import Random
from typeguard import typechecked


class EpsilonGreedy(ExplorationStrategy):
    @typechecked
    def __init__(self, action_space: Space, init_random_prob: float, decay_type: (int, None),
                 random_state: Random = Random()):
        self.action_space = action_space
        self.init_random_prob = init_random_prob
        self.decay_type = decay_type
        # todo decay for the prob
        self.random_state = random_state
        super(ExplorationStrategy, self).__init__()

    def predict(self, **kwargs):
        if self.random_state.unwrapped().random() < self.init_random_prob:
            return self.action_space.sample()
        else:
            algo = kwargs.pop('algo')
            return algo.predict(**kwargs)
