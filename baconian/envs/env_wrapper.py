from baconian.core.core import Env
from baconian.common.spaces import Box
from baconian.envs.gym_env import GymEnv
import numpy as np


class Wrapper(Env):

    def __init__(self, env: Env):
        if isinstance(env, GymEnv):
            self.env = env.unwrapped_gym
            self.src_env = env
        else:
            self.env = env
            self.src_env = env
        super().__init__(name=env.name + '_wrapper', copy_from_env=env)

    def __getattr__(self, item):
        if hasattr(self.src_env, item):
            return getattr(self.src_env, item)
        if hasattr(self.env, item):
            return getattr(self.env, item)
        raise AttributeError()

    def seed(self, seed=None):
        return self.src_env.seed(seed)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def spec(self):
        return self.env.spec

    @classmethod
    def class_name(cls):
        return cls.__name__

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    def reset(self):
        return self.src_env.reset()

    def get_state(self):
        return self.src_env.get_state()


class ObservationWrapper(Wrapper):
    def reset(self):
        observation = self.src_env.reset()
        return self._observation(observation)

    def step(self, action):
        observation, reward, done, info = self.src_env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return self._observation(observation)

    def _observation(self, observation):
        raise NotImplementedError

    def get_state(self):
        return self._observation(self.src_env.get_state())


class RewardWrapper(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.src_env.step(action)
        return observation, self.reward(observation, action, reward, done, info), done, info

    def reward(self, observation, action, reward, done, info):
        return self._reward(observation, action, reward, done, info)

    def _reward(self, observation, action, reward, done, info):
        raise NotImplementedError


class ActionWrapper(Wrapper):
    def step(self, action):
        action = self.action(action)
        return self.src_env.step(action)

    def action(self, action):
        return self._action(action)

    def _action(self, action):
        raise NotImplementedError

    def reverse_action(self, action):
        return self._reverse_action(action)

    def _reverse_action(self, action):
        raise NotImplementedError


class StepObservationWrapper(ObservationWrapper):
    def __init__(self, env: Env, step_limit=100000):
        super().__init__(env=env)
        assert isinstance(self.src_env.observation_space, Box), 'not support non Box space for step observation wrapper'
        self.src_env.observation_space = Box(low=np.concatenate([self.src_env.observation_space.low, np.array([0])]),
                                             high=np.concatenate(
                                                 [self.src_env.observation_space.high, np.array([step_limit])]))
        self.src_env.env_spec.obs_space = self.src_env.observation_space
        self.observation_space = self.src_env.observation_space

    def _observation(self, observation):
        obs = np.array(observation)
        return np.concatenate([obs, np.array([self.src_env.trajectory_level_step_count])])
