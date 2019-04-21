from baconian.core.core import Basic, EnvSpec
from baconian.core.status import StatusWithSubInfo
import abc
from typeguard import typechecked
from baconian.common.logging import Recorder


class Algo(Basic):
    STATUS_LIST = ['NOT_INIT', 'JUST_INITED', 'TRAIN', 'TEST']
    INIT_STATUS = 'NOT_INIT'

    @typechecked
    def __init__(self, env_spec: EnvSpec, name: str = 'algo'):
        super().__init__(status=StatusWithSubInfo(obj=self), name=name)
        self.env_spec = env_spec
        self.recorder = Recorder()

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

    @property
    def is_training(self):
        """

        :return:
        """
        return self.get_status()['status'] == 'TRAIN'

    @property
    def is_testing(self):
        """

        :return:
        """
        return self.get_status()['status'] == 'TEST'
