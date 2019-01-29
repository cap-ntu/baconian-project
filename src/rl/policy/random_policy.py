from gym import Space

from src.envs.env_spec import EnvSpec
from src.rl.policy.policy import Policy
from typeguard import typechecked
from overrides.overrides import overrides


class UniformRandomPolicy(Policy):

    @typechecked
    def __init__(self, env_spec: EnvSpec):
        super().__init__(env_spec)

    @overrides
    def sample(self, obs, **kwargs):
        return self.action_space.sample()
