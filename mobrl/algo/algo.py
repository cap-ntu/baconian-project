from mobrl.core.core import Basic, EnvSpec
from mobrl.core.status import StatusWithSingleInfo, StatusWithSubInfo
import abc
from typeguard import typechecked
from mobrl.common.util.logging import Logger, Recorder


# import numpy as np


class Algo(Basic):
    STATUS_LIST = ['NOT_INIT', 'JUST_INITED', 'TRAIN', 'TEST']
    INIT_STATUS = 'NOT_INIT'

    @typechecked
    def __init__(self, env_spec: EnvSpec, name: str = 'algo'):
        self.env_spec = env_spec
        self.recorder = Recorder()
        super().__init__(status=StatusWithSubInfo(obj=self), name=name)

    def init(self):
        self._status.set_status('JUST_INITED')

    def train(self, *arg, **kwargs) -> dict:
        self._status.set_status('TRAIN')
        return dict()

    def test(self, *arg, **kwargs) -> dict:
        self._status.set_status('TEST')
        return dict()

    @abc.abstractmethod
    def predict(self, *arg, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def append_to_memory(self, *args, **kwargs):
        raise NotImplementedError

    # def register_logger(self, logger: Logger):
    #     raise NotImplementedError
