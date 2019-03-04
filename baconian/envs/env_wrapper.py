from baconian.core.core import Env
from baconian.common.util.logging import ConsoleLogger
from baconian.envs.gym_env import GymEnv
import gym.error as error
from gym.core import Wrapper as gym_wrapper


class Wrapper(gym_wrapper):

    def __init__(self, env: Env):
        if isinstance(env, GymEnv):
            self.env = env.unwrapped_gym
            self.src_env = env
        else:
            self.env = env
            self.src_env = env
        gym_wrapper.__init__(self, env=self.env)


class ObservationWrapper(Wrapper):
    def _reset(self):
        observation = self.env.reset()
        return self._observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return self._observation(observation)

    def _observation(self, observation):
        raise NotImplementedError


class RewardWrapper(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(observation, action, reward, done, info), done, info

    def reward(self, observation, action, reward, done, info):
        return self._reward(observation, action, reward, done, info)

    def _reward(self, observation, action, reward, done, info):
        raise NotImplementedError


class ActionWrapper(Wrapper):
    def step(self, action):
        action = self.action(action)
        return self.env.step(action)

    def action(self, action):
        return self._action(action)

    def _action(self, action):
        raise NotImplementedError

    def reverse_action(self, action):
        return self._reverse_action(action)

    def _reverse_action(self, action):
        raise NotImplementedError
