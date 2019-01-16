from src.core.basic import Basic
import abc
from src.envs.env_spec import EnvSpec


class Algo(Basic, abc.ABC):
    def __init__(self, env_spec: EnvSpec):
        self.env_spec = env_spec
        super().__init__()

    @abc.abstractmethod
    def init_op(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, *arg, **kwargs):
        pass

    @abc.abstractmethod
    def test(self, *arg, **kwargs):
        pass
