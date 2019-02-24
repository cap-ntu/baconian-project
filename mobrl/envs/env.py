import gym
from mobrl.core.basic import Basic
from typeguard import typechecked
from mobrl.common.util.recorder import Recorder
from mobrl.core.status import StatusWithSingleInfo, register_counter_info_to_status_decorator


class Env(gym.Env, Basic):
    """
    Abstract class for environment
    """
    key_list = []
    STATUS_LIST = ['JUST_RESET', 'JUST_INITED', 'STEPPING', 'NOT_INITED']
    INIT_STATUS = 'NOT_INITED'

    @typechecked
    def __init__(self):
        super(Env, self).__init__(status=StatusWithSingleInfo(obj=self))
        self.action_space = None
        self.observation_space = None
        self.step_count = None
        self.recorder = Recorder

    @register_counter_info_to_status_decorator(increment=1, info_key='step', under_status='STEPPING')
    def step(self, action):
        self._status.set_status('STEPPING')

    @register_counter_info_to_status_decorator(increment=1, info_key='reset', under_status='JUST_RESET')
    def reset(self):
        self._status.set_status('JUST_RESET')

    @register_counter_info_to_status_decorator(increment=1, info_key='init', under_status='JUST_INITED')
    def init(self):
        self._status.set_status('JUST_INITED')


if __name__ == '__main__':
    pass
