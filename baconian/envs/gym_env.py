from baconian.core.core import Env, EnvSpec
import gym.envs
from gym.envs.registration import registry

have_mujoco_flag = True
try:
    from gym.envs.mujoco import mujoco_env
except Exception:
    have_mujoco_flag = False

import numpy as np
import types
from gym.spaces import *
from gym.spaces import Space as GymSpace
import baconian.common.spaces as garage_space
from typeguard import typechecked
from baconian.core.status import register_counter_info_to_status_decorator

_env_inited_count = dict()


def make(gym_env_id, allow_multiple_env=True):
    """

    :param gym_env_id:
    :param allow_multiple_env:
    :return:
    """
    if allow_multiple_env is True:

        if gym_env_id not in _env_inited_count:
            _env_inited_count[gym_env_id] = 0
        else:
            _env_inited_count[gym_env_id] += 1

        return GymEnv(gym_env_id, name='{}_{}'.format(gym_env_id, _env_inited_count[gym_env_id]))
    else:
        return GymEnv(gym_env_id)


@typechecked
def space_converter(space: GymSpace):
    """

    :param space:
    :return:
    """
    if isinstance(space, Box):
        return garage_space.Box(low=space.low, high=space.high)
    # elif isinstance(space, Dict):
    #     return garage_space.Dict(space.spaces)
    elif isinstance(space, Discrete):
        return garage_space.Discrete(space.n)
    elif isinstance(space, Tuple):
        return garage_space.Tuple(list(map(space_converter, space.spaces)))
    else:
        raise NotImplementedError


class GymEnv(Env):
    """
    Gym environment wrapping module
    """
    _all_gym_env_id = list(registry.env_specs.keys())

    def __init__(self, gym_env_id: str, name: str = None):
        """

        :param gym_env_id:
        :param name:
        """
        super().__init__(name=name if name else gym_env_id)
        self.env_id = gym_env_id
        if gym_env_id not in self._all_gym_env_id:
            raise ValueError('Env id: {} is not supported currently'.format(gym_env_id))
        self._gym_env = gym.make(gym_env_id)
        self.action_space = space_converter(self._gym_env.action_space)
        self.observation_space = space_converter(self._gym_env.observation_space)
        if isinstance(self.action_space, garage_space.Box):
            self.action_space.low = np.nan_to_num(self.action_space.low)
            self.action_space.high = np.nan_to_num(self.action_space.high)
            self.action_space.sample = types.MethodType(self._sample_with_nan, self.action_space)
        if isinstance(self.observation_space, garage_space.Box):
            self.observation_space.low = np.nan_to_num(self.observation_space.low)
            self.observation_space.high = np.nan_to_num(self.observation_space.high)
            self.observation_space.sample = types.MethodType(self._sample_with_nan, self.observation_space)
        self.env_spec = EnvSpec(obs_space=self.observation_space,
                                action_space=self.action_space)

        self.reward_range = self._gym_env.reward_range

    @register_counter_info_to_status_decorator(increment=1, info_key='step', under_status=('TRAIN', 'TEST'),
                                               ignore_wrong_status=True)
    def step(self, action):
        """

        :param action:
        :return:
        """
        super().step(action)
        return self.unwrapped.step(action=action)

    def reset(self):
        """

        :return:
        """
        super().reset()

        return self.unwrapped.reset()

    def init(self):
        """

        :return:
        """
        super().init()

        return self.reset()

    def seed(self, seed=None):
        """

        :param seed:
        :return:
        """
        return super().seed(seed)

    def get_state(self):
        """

        :return:
        """
        if (have_mujoco_flag is True and isinstance(self.unwrapped_gym, mujoco_env.MujocoEnv)) or (
                hasattr(self.unwrapped_gym, '_get_obs') and callable(self.unwrapped_gym._get_obs)):
            return self.unwrapped_gym._get_obs()
        elif hasattr(self.unwrapped_gym, '_get_ob') and callable(self.unwrapped_gym._get_ob):
            return self.unwrapped_gym._get_ob()
        elif hasattr(self.unwrapped_gym, 'state'):
            return self.unwrapped_gym.state if isinstance(self.unwrapped_gym.state, np.ndarray) else np.array(
                self.unwrapped_gym.state)
        elif hasattr(self.unwrapped_gym, 'observation'):
            return self.unwrapped_gym.observation if isinstance(self.unwrapped_gym.observation,
                                                                np.ndarray) else np.array(
                self.unwrapped_gym.state)
        else:
            raise ValueError('Env id: {} is not supported for method get_state'.format(self.env_id))

    @property
    def unwrapped(self):
        """

        :return:
        """
        return self._gym_env

    @property
    def unwrapped_gym(self):
        """

        :return:
        """
        if hasattr(self._gym_env, 'unwrapped'):
            return self._gym_env.unwrapped
        else:
            return self._gym_env

    @staticmethod
    def _sample_with_nan(space: garage_space.Space):
        """

        :param space:
        :return:
        """
        assert isinstance(space, garage_space.Box)
        high = np.ones_like(space.low)
        low = -1 * np.ones_like(space.high)
        return np.clip(np.random.uniform(low=low, high=high, size=space.low.shape),
                       a_min=space.low,
                       a_max=space.high)
