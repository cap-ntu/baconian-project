from src.core.basic import Basic
import abc
from src.envs.env_spec import EnvSpec
from src.rl.algo.model_based.models.dynamics_model import DynamicsModel
from typeguard import typechecked


class Algo(Basic, abc.ABC):
    STATUS_LIST = ['NOT_INIT', 'JUST_INITED', 'TRAIN', 'TEST']
    INIT_STATUS = 'NOT_INIT'

    @typechecked
    def __init__(self, env_spec: EnvSpec):
        self.env_spec = env_spec
        super().__init__()

    def init(self):
        self.status.set_status('JUST_INITED')

    def train(self, *arg, **kwargs) -> dict:
        self.status.set_status('TRAIN')

    def test(self, *arg, **kwargs):
        self.status.set_status('TEST')

    def predict(self, *arg, **kwargs):
        raise NotImplementedError

    def append_to_memory(self, *args, **kwargs):
        raise NotImplementedError


class ModelFreeAlgo(Algo):
    pass


class ModelBasedAlgo(Algo):
    def __init__(self, env_spec, dynamics_model: DynamicsModel):
        super().__init__(env_spec)
        self.dynamics_model = dynamics_model
