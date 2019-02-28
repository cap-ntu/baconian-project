import gym
import numpy as np
from typeguard import typechecked

from mobrl.common.spaces import Space
from mobrl.common.special import flat_dim, flatten

from mobrl.common.util.logging import Recorder
from mobrl.config.global_config import GlobalConfig
from mobrl.core.status import StatusWithSingleInfo, register_counter_info_to_status_decorator, Status
from mobrl.core.util import register_name_globally, init_func_arg_record_decorator


class Basic(object):
    """ Basic class within the whole framework"""
    STATUS_LIST = GlobalConfig.DEFAULT_BASIC_STATUS_LIST
    INIT_STATUS = GlobalConfig.DEFAULT_BASIC_INIT_STATUS
    required_key_dict = ()

    def __init__(self, name: str, status=None):
        if not status:
            self._status = Status(self)
        else:
            self._status = status
        self._name = name
        register_name_globally(name=name, obj=self)

    def init(self):
        raise NotImplementedError

    def get_status(self) -> dict:
        return self._status.get_status()

    def set_status(self, val):
        self._status.set_status(val)

    @property
    def name(self):
        return self._name

    @property
    def status_list(self):
        return self.STATUS_LIST

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError


class Env(gym.Env, Basic):
    """
    Abstract class for environment
    """
    key_list = []
    STATUS_LIST = ['JUST_RESET', 'JUST_INITED', 'STEPPING', 'NOT_INITED']
    INIT_STATUS = 'NOT_INITED'

    @typechecked
    def __init__(self, name='env'):
        super(Env, self).__init__(status=StatusWithSingleInfo(obj=self), name=name)
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


class EnvSpec(object):
    @init_func_arg_record_decorator()
    @typechecked
    def __init__(self, obs_space: Space, action_space: Space):
        self._obs_space = obs_space
        self._action_space = action_space
        self.obs_shape = tuple(np.array(self.obs_space.sample()).shape)
        if len(self.obs_shape) == 0:
            self.obs_shape = (1,)
        self.action_shape = tuple(np.array(self.action_space.sample()).shape)
        if len(self.action_shape) == 0:
            self.action_shape = ()

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def flat_obs_dim(self) -> int:
        return flat_dim(self.obs_space)

    @property
    def flat_action_dim(self) -> int:
        return flat_dim(self.action_space)

    @staticmethod
    def flat(space: Space, obs_or_action: (np.ndarray, list)):
        return flatten(space, obs_or_action)

    def flat_action(self, action: (np.ndarray, list)):
        return flatten(self.action_space, action)

    def flat_obs(self, obs: (np.ndarray, list)):
        return flatten(self.obs_space, obs)
