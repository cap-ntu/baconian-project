from mbrl.core.basic import Basic
import abc
from mbrl.envs.env_spec import EnvSpec
from mbrl.rl.algo.model_based.models.dynamics_model import DynamicsModel
from typeguard import typechecked


class Algo(Basic, abc.ABC):
    STATUS_LIST = ['NOT_INIT', 'JUST_INITED', 'TRAIN', 'TEST']
    INIT_STATUS = 'NOT_INIT'

    @typechecked
    def __init__(self, env_spec: EnvSpec, name: str = 'algo'):
        self.env_spec = env_spec
        self.name = name
        super().__init__()

    def init(self):
        self.status.set_status('JUST_INITED')

    def train(self, *arg, **kwargs) -> dict:
        self.status.set_status('TRAIN')
        return dict()

    def test(self, *arg, **kwargs) -> dict:
        self.status.set_status('TEST')
        return dict()

    def predict(self, *arg, **kwargs):
        raise NotImplementedError

    def append_to_memory(self, *args, **kwargs):
        raise NotImplementedError


class ModelFreeAlgo(Algo):
    def __init__(self, env_spec: EnvSpec, name: str = 'model_free_algo'):
        super(ModelFreeAlgo, self).__init__(env_spec, name)


class ModelFreeOnPolicyAlgo(ModelFreeAlgo):
    pass


class ModelFreeOffPolicyAlgo(ModelFreeAlgo):
    pass


class OnPolicyAlgo(Algo):
    pass


class OffPolicyAlgo(Algo):
    pass


class ValueBasedAlgo(Algo):
    pass


class PolicyBasedAlgo(Algo):
    pass


class ModelBasedAlgo(Algo):
    def __init__(self, env_spec, dynamics_model: DynamicsModel, name: str = 'model_based_algo'):
        super(ModelBasedAlgo, self).__init__(env_spec, name)
        self.dynamics_model = dynamics_model
