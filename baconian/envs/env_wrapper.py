from baconian.core.core import Env
from baconian.common.util.logging import ConsoleLogger
from baconian.envs.gym_env import GymEnv
from typeguard import typechecked
import gym.error as error


class Wrapper(Env):
    # Clear metadata so by default we don't override any keys.
    metadata = {}

    _owns_render = False

    # Make sure self.env is always defined, even if things break
    # early.
    env = None

    def __init__(self, env: Env):
        Env.__init__(self, name='wrapped_{}'.format(env.name))
        if isinstance(env, GymEnv):
            self.env = env.unwrapped_gym
            self.src_env = env
        else:
            self.env = env
            self.src_env = env
        # Merge with the base metadata
        metadata = self.metadata
        self.metadata = self.env.metadata.copy()
        self.metadata.update(metadata)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self._spec = self.env.spec
        self._unwrapped = self.env.unwrapped
        self.step_count = self.src_env.step_count
        self.recorder = self.src_env.recorder

        self._update_wrapper_stack()
        if env and env._configured:
            ConsoleLogger().print('WARNING', "Attempted to wrap env %s after .configure() was called.", env)

    def _update_wrapper_stack(self):
        """
        Keep a list of all the wrappers that have been appended to the stack.
        """
        self._wrapper_stack = getattr(self.env, '_wrapper_stack', [])
        self._check_for_duplicate_wrappers()
        self._wrapper_stack.append(self)

    def _check_for_duplicate_wrappers(self):
        """Raise an error if there are duplicate wrappers. Can be overwritten by subclasses"""
        if self.class_name() in [wrapper.class_name() for wrapper in self._wrapper_stack]:
            raise error.DoubleWrapperError("Attempted to double wrap with Wrapper: {}".format(self.class_name()))

    @classmethod
    def class_name(cls):
        return cls.__name__

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode='human', close=False):
        if self.env is None:
            return
        return self.env.render(mode, close)

    def close(self):
        if self.env is None:
            return
        return self.env.close()

    def _configure(self, *args, **kwargs):
        return self.env.configure(*args, **kwargs)

    def _seed(self, seed=None):
        return self.env.seed(seed)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def spec(self):
        if self._spec is None:
            self._spec = self.env.spec
        return self._spec

    @spec.setter
    def spec(self, spec):
        # Won't have an env attr while in the __new__ from gym.Env
        if self.env is not None:
            self.env.spec = spec
        self._spec = spec


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
