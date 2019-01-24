from src.core.basic import Basic
from src.common.sampler.sampler import Sampler
from src.core.global_config import GlobalConfig
from src.rl.algo.algo import Algo
from typeguard import typechecked
from src.rl.exploration_strategy.base import ExplorationStrategy
from src.common.sampler.sample_data import SampleData
from src.envs.env import Env


class Agent(Basic):

    @typechecked
    def __init__(self, env: GlobalConfig.DEFAULT_ALLOWED_GYM_ENV_TYPE + (Env,), algo: Algo,
                 sampler: Sampler = Sampler(),
                 exploration_strategy=None):
        super(Agent, self).__init__()
        self.env = env
        self.algo = algo
        self._env_step_count = 0
        self.sampler = sampler
        if exploration_strategy:
            assert isinstance(exploration_strategy, ExplorationStrategy)
            self.explorations_strategy = exploration_strategy

    @property
    def env_sample_count(self):
        return self._env_step_count

    @env_sample_count.setter
    def env_sample_count(self, new_value):
        assert isinstance(new_value, int) and new_value >= 0
        self._env_step_count = new_value

    def predict(self, in_test_flag, **kwargs):
        if self.explorations_strategy and not in_test_flag:
            return self.explorations_strategy.predict(**kwargs, algo=self.algo)
        else:
            return self.algo.predict(**kwargs)

    def sample(self, env, sample_count: int, in_test_flag: bool, store_flag=False):
        batch_data = self.sampler.sample(agent=self,
                                         env=env,
                                         in_test_flag=in_test_flag,
                                         sample_count=sample_count)
        if store_flag is True:
            self.store_samples(samples=batch_data)
        return batch_data

    def init(self):
        self.algo.init()
        print("%s init finished" % type(self).__name__)

    @typechecked
    def store_samples(self, samples: SampleData):
        self.algo.append_to_memory(samples)

    def update(self) -> dict:
        return self.algo.train()

    def reset(self):
        raise NotImplementedError
