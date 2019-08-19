from baconian.core.core import Env, EnvSpec

have_mujoco_flag = True
try:
    from dm_control import mujoco
    from gym.envs.mujoco import mujoco_env
    from dm_control import suite
    from dm_control.rl.specs import ArraySpec
    from dm_control.rl.specs import BoundedArraySpec
    from collections import OrderedDict
except Exception:
    have_mujoco_flag = False

import numpy as np
import types
from gym.spaces import *
import baconian.common.spaces as garage_space


def convert_dm_control_to_gym_space(dm_control_space):
    r"""Convert dm_control space to gym space. """
    if isinstance(dm_control_space, BoundedArraySpec):
        space = Box(low=dm_control_space.minimum,
                    high=dm_control_space.maximum,
                    dtype=dm_control_space.dtype)
        assert space.shape == dm_control_space.shape
        return garage_space.Box(low=space.low, high=space.high)
    elif isinstance(dm_control_space, ArraySpec) and not isinstance(dm_control_space, BoundedArraySpec):
        space = Box(low=-float('inf'),
                    high=float('inf'),
                    shape=dm_control_space.shape,
                    dtype=dm_control_space.dtype)
        return garage_space.Box(low=space.low, high=space.high)
    elif isinstance(dm_control_space, OrderedDict):
        space = Dict(OrderedDict([(key, convert_dm_control_to_gym_space(value))
                                  for key, value in dm_control_space.items()]))
        return garage_space.Dict(space.spaces)
    else:
        raise NotImplementedError


_env_inited_count = dict()


# def make(gym_env_id, allow_multiple_env=True):
#     """
#
#     :param gym_env_id:
#     :param allow_multiple_env:
#     :return:
#     """
#     if allow_multiple_env is True:
#
#         if gym_env_id not in _env_inited_count:
#             _env_inited_count[gym_env_id] = 0
#         else:
#             _env_inited_count[gym_env_id] += 1
#
#         return GymEnv(gym_env_id, name='{}_{}'.format(gym_env_id, _env_inited_count[gym_env_id]))
#     else:
#         return GymEnv(gym_env_id)


class DMControlEnv(Env):
    """
    DeepMind Control Suite environment wrapping module
    """

    def __init__(self, dmcs_env_id: str, name: str = None):
        """

        :param dmcs_env_id:
        :param name:
        """
        super().__init__(name=name if name else dmcs_env_id)
        self.env_id = dmcs_env_id
        self.timestep = {}
        try:
            self.env = suite.load(dmcs_env_id, name)
        except ValueError:
            raise ValueError('Env id: {} and task: {} is not supported currently'.format(dmcs_env_id, name))

        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': int(np.round(1.0 / self.env.control_timestep()))}

        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())
        self.observation_space = convert_dm_control_to_gym_space(self.env.observation_spec())
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

        self.viewer = None

    def step(self, action):
        """

        :param action:
        :return:
        """
        super().step(action)
        self.timestep = self.env.step(action)
        observation = self.timestep.observation
        reward = self.timestep.reward
        done = self.timestep.last()
        info = {}
        return observation, reward, done, info

    def reset(self):
        """

        :return:
        """
        super().reset()

        return self.env.reset()

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
        return self.env.task.random.seed(seed)

    def get_state(self):
        """

        :return:
        """
        if self.timestep != {}:
            return self.timestep.observation
        else:
            raise ValueError('Env id: {} does not have an observation yet.'.format(self.env_id))

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


if __name__ == '__main__':
    a = suite.load("cartpole", "swingup")
    a = DMControlEnv("cartpole", "swingup")
