from mobrl.config.dict_config import DictConfig
from typeguard import typechecked
from transitions import Machine
from mobrl.common.util.logging import Recorder
from mobrl.core.status import StatusCollector, StatusWithSingleInfo
from mobrl.core.core import Basic


class Pipeline(Basic):
    REQUIRED_KEY_LIST = []
    STATE_LIST = []
    INIT_STATE = None

    @typechecked
    def __init__(self, config: DictConfig, init_state: str, states: list, transitions: (list, dict), name='pipeline'):
        self.config = config
        self.finite_state_machine = Machine(model=self, transitions=transitions, states=states, initial=init_state)
        self.total_test_samples = 0
        self.total_train_samples = 0
        self.recorder = Recorder(flush_by_split_status=False)
        self.status_collector = StatusCollector()
        Basic.__init__(self, name=name, status=StatusWithSingleInfo(obj=self))

        for state_name in self.STATE_LIST:
            if not hasattr(self, 'on_enter_{}'.format(state_name)):
                raise AssertionError('{} method is missed'.format('on_enter_{}'.format(state_name)))
            if not hasattr(self, 'on_exit_{}'.format(state_name)):
                raise AssertionError('{} method is missed'.format('on_exit_{}'.format(state_name)))

    def launch(self, *args, **kwargs) -> bool:
        raise NotImplementedError
