from src.core.basic import Basic
import abc
from src.core.global_config import GlobalConfig
from src.envs.env_spec import EnvSpec
from src.envs.env import Env
from src.envs.model_based_env import ModelBasedEnv
from typeguard import typechecked
from src.common.sampler.sampler import Sampler


class Algo(Basic, abc.ABC):
    STATUS_LIST = ['NOT_INIT', 'JUST_INITED', 'TRAIN', 'TEST']
    INIT_STATUS = 'NOT_INIT'

    @typechecked
    def __init__(self, env: GlobalConfig.DEFAULT_ALLOWED_GYM_ENV_TYPE + (Env,)):
        self.env_spec = EnvSpec(obs_space=env.observation_space, action_space=env.action_space)
        self.env = env
        super().__init__()

    def init(self):
        self.status.set_status('JUST_INITED')

    @typechecked
    def train(self, *arg, **kwargs) -> dict:
        self.status.set_status('TRAIN')
        return dict()

    def test(self, *arg, **kwargs):
        self.status.set_status('TEST')

    def predict(self, *arg, **kwargs):
        raise NotImplementedError

    def append_to_memory(self, *args, **kwargs):
        raise NotImplementedError


class ModelFreeAlgo(Algo):

    def __init__(self, env, sampler: Sampler = Sampler()):
        super().__init__(env)
        self.sampler = sampler


class ModelBasedAlgo(Algo):
    def __init__(self, env, dynamics_model: ModelBasedEnv, sampler: Sampler = Sampler()):
        super().__init__(env)
        self.dynamics_model = dynamics_model
        self.sampler = sampler
