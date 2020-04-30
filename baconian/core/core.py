import gym
import numpy as np
from typeguard import typechecked

from baconian.common.spaces import Space
from baconian.common.special import flat_dim, flatten

from baconian.common.logging import Recorder
from baconian.config.global_config import GlobalConfig
from baconian.core.status import *
from baconian.core.util import register_name_globally, init_func_arg_record_decorator

"""
This module contains the some core classes of baconian
"""


class Basic(object):
    """ Basic class within the whole framework"""
    STATUS_LIST = GlobalConfig().DEFAULT_BASIC_STATUS_LIST
    INIT_STATUS = GlobalConfig().DEFAULT_BASIC_INIT_STATUS
    required_key_dict = ()
    allow_duplicate_name = False

    def __init__(self, name: str, status=None):
        """
        Init a new Basic instance.

        :param name: name of the object, can be determined to generate log path, handle tensorflow name scope etc.
        :type name: str
        :param status: A status instance :py:class:`~baconian.core.status.Status` to indicate the status of the object
        :type status: Status
        """

        if not status:
            self._status = Status(self)
        else:
            self._status = status
        self._name = name
        register_name_globally(name=name, obj=self)

    def init(self, *args, **kwargs):
        """Initialize the object"""
        raise NotImplementedError

    def get_status(self) -> dict:
        """ Return the object's status, a dictionary."""
        return self._status.get_status()

    def set_status(self, val):
        """ Set the object's status."""
        self._status.set_status(val)

    @property
    def name(self):
        """ The name(id) of object, a string."""
        return self._name

    @property
    def status_list(self):
        """ Status list of the object, ('TRAIN', 'TEST')."""
        return self.STATUS_LIST

    def save(self, *args, **kwargs):
        """ Save the parameters in training checkpoints."""
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """ Load the parameters from training checkpoints."""
        raise NotImplementedError


class Env(gym.Env, Basic):
    """
    Abstract class for environment
    """
    key_list = ()
    STATUS_LIST = ('JUST_RESET', 'INITED', 'TRAIN', 'TEST', 'CREATED')
    INIT_STATUS = 'CREATED'

    @typechecked
    def __init__(self, name: str = 'env', copy_from_env=None):
        super(Env, self).__init__(status=StatusWithSubInfo(obj=self), name=name)
        self.action_space = None
        self.observation_space = None
        self.trajectory_level_step_count = 0
        self.recorder = Recorder(default_obj=self)
        self._last_reset_point = 0
        self.total_step_count_fn = lambda: self._status.group_specific_info_key(info_key='step', group_way='sum')
        self.env_spec = None
        if copy_from_env:
            assert isinstance(copy_from_env, Env)
            self.action_space = copy_from_env.action_space
            self.observation_space = copy_from_env.observation_space
            self.trajectory_level_step_count = copy_from_env.trajectory_level_step_count
            self.trajectory_level_step_count = copy_from_env._last_reset_point
            self.env_spec = copy_from_env.env_spec

    @register_counter_info_to_status_decorator(increment=1, info_key='step', under_status=('TRAIN', 'TEST'),
                                               ignore_wrong_status=True)
    def step(self, action):
        """

        :param action: agent's action, the environment will react responding to action
        :type action: method
        """
        self.trajectory_level_step_count += 1
        pass

    @register_counter_info_to_status_decorator(increment=1, info_key='reset', under_status='JUST_RESET')
    def reset(self):
        """ Set the status to 'JUST_RESET', and update new reset point"""
        self._status.set_status('JUST_RESET')
        self._last_reset_point = self.total_step_count_fn()
        self.trajectory_level_step_count = 0

    @register_counter_info_to_status_decorator(increment=1, info_key='init', under_status='INITED')
    def init(self):
        """ Set the status to 'INITED'."""
        self._status.set_status('INITED')

    def get_state(self):
        """ Get the status of the environment."""
        raise NotImplementedError

    def seed(self, seed=None):
        """

        :param seed: seed to generate random number
        :type seed: int
        :return: seed of the unwrapped environment
        :rtype: int
        """
        return self.unwrapped.seed(seed=seed)


class EnvSpec(object):
    @init_func_arg_record_decorator()
    @typechecked
    def __init__(self, obs_space: Space, action_space: Space):
        self._obs_space = obs_space
        self._action_space = action_space

    @property
    def obs_shape(self):
        obs_shape = tuple(np.array(self.obs_space.sample()).shape)
        if len(obs_shape) == 0:
            obs_shape = (1,)
        return obs_shape

    @property
    def action_shape(self):
        action_shape = tuple(np.array(self.action_space.sample()).shape)
        if len(action_shape) == 0:
            action_shape = ()
        return action_shape

    @property
    def obs_space(self):
        """

        :return: Observation space of environment
        :rtype: Space
        """
        return self._obs_space

    @property
    def action_space(self):
        """

        :return: Action space of environment
        :rtype: Space
        """
        return self._action_space

    @obs_space.setter
    def obs_space(self, s: Space):
        self._obs_space = s

    @action_space.setter
    def action_space(self, s: Space):
        self._action_space = s

    @property
    def flat_obs_dim(self) -> int:
        """

        :return: the dimension(length) of flatten observation space
        :rtype: int
        """
        return int(flat_dim(self.obs_space))

    @property
    def flat_action_dim(self) -> int:
        """

        :return: the dimension(length) of flatten action space
        :rtype: int
        """
        return int(flat_dim(self.action_space))

    @staticmethod
    def flat(space: Space, obs_or_action: (np.ndarray, list)):
        """
        flat the input obs or action
        :param space: space of environment
        :type space: Space
        :param obs_or_action: action or observation space
        :type obs_or_action: (np.ndarray, list)
        :return: flatten action or observation space
        :rtype: Space
        """
        return flatten(space, obs_or_action)

    def flat_action(self, action: (np.ndarray, list)):
        """

        :param action: action taken by agent
        :type action: (np.ndarray, list)
        :return: flatten action parameter
        :rtype: np.ndarray
        """
        return flatten(self.action_space, action)

    def flat_obs(self, obs: (np.ndarray, list)):
        """

        :param obs: observation of the agent
        :type obs:  (np.ndarray, list)
        :return: flatten observation parameter
        :rtype: np.ndarray
        """
        return flatten(self.obs_space, obs)
