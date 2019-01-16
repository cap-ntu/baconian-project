from gym.core import Space
from typeguard import typechecked
from src.envs.util import *


class EnvSpec(object):

    @typechecked
    def __init__(self, obs_space: Space, action_space: Space):
        self._obs_space = obs_space
        self._action_space = action_space

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
