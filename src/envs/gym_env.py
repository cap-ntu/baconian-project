from src.envs.env import Env
import gym.envs
from gym.envs.registration import registry
from gym.envs.mujoco import mujoco_env
import numpy as np
from gym.spaces.box import Box
from gym.core import Space
import types


def make(gym_env_id):
    return GymEnv(gym_env_id)


class GymEnv(Env):
    _all_gym_env_id = list(registry.env_specs.keys())

    def __init__(self, gym_env_id: str):
        super().__init__()
        self.env_id = gym_env_id
        if gym_env_id not in self._all_gym_env_id:
            raise ValueError('Env id: {} is not supported currently'.format(gym_env_id))
        self._gym_env = gym.make(gym_env_id)
        self.action_space = self._gym_env.action_space
        self.observation_space = self._gym_env.observation_space
        if isinstance(self.action_space, Box):
            self.action_space.low = np.nan_to_num(self.action_space.low)
            self.action_space.high = np.nan_to_num(self.action_space.high)
            self.action_space.sample = types.MethodType(self._sample_with_nan, self.action_space)
        if isinstance(self.observation_space, Box):
            self.observation_space.low = np.nan_to_num(self.observation_space.low)
            self.observation_space.high = np.nan_to_num(self.observation_space.high)
            self.observation_space.sample = types.MethodType(self._sample_with_nan, self.observation_space)

        self.reward_range = self._gym_env.reward_range

    def step(self, action):
        return self.unwrapped.step(action=action)

    def reset(self):
        return self.unwrapped.reset()

    def init(self):
        return self.reset()

    def seed(self, seed=None):
        return super().seed(seed)

    def get_state(self):
        if isinstance(self.unwrapped_gym, mujoco_env.MujocoEnv) or (
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
        return self._gym_env

    @property
    def unwrapped_gym(self):
        if hasattr(self._gym_env, 'unwrapped'):
            return self._gym_env.unwrapped
        else:
            return self._gym_env

    @staticmethod
    def _sample_with_nan(space: Space):
        assert isinstance(space, Box)
        high = np.ones_like(space.low)
        low = -1 * np.ones_like(space.high)
        return np.clip(np.random.uniform(low=low, high=high, size=space.low.shape),
                       a_min=space.low,
                       a_max=space.high)
