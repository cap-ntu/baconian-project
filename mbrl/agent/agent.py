from mbrl.core.basic import Basic
from mbrl.common.sampler.sampler import Sampler
from mbrl.config.global_config import GlobalConfig
from mbrl.algo.algo import Algo
from typeguard import typechecked
from mbrl.algo.rl.misc.exploration_strategy.base import ExplorationStrategy
from mbrl.common.sampler.sample_data import SampleData
from mbrl.envs.env import Env
from mbrl.envs.env_spec import EnvSpec


class Agent(Basic):

    @typechecked
    def __init__(self, env: GlobalConfig.DEFAULT_ALLOWED_GYM_ENV_TYPE + (Env,), algo: Algo, env_spec: EnvSpec,
                 sampler: Sampler = None,
                 exploration_strategy=None):
        super(Agent, self).__init__()
        self.env = env
        self.algo = algo
        self._env_step_count = 0
        self.sampler = sampler
        if exploration_strategy:
            assert isinstance(exploration_strategy, ExplorationStrategy)
            self.explorations_strategy = exploration_strategy
        self.sampler = sampler if sampler else Sampler(env_spec=env_spec)

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
        self.algo.append_to_memory(samples=samples)

    def update(self, **kwargs) -> dict:
        return self.algo.train(**kwargs)

    def reset(self):
        raise NotImplementedError
