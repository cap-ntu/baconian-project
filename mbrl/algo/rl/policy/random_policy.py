from mbrl.envs.env_spec import EnvSpec
from mbrl.algo.rl.policy.policy import Policy
from typeguard import typechecked
from overrides.overrides import overrides


class UniformRandomPolicy(Policy):

    @typechecked
    def __init__(self, env_spec: EnvSpec):
        super().__init__(env_spec)

    @overrides
    def forward(self, obs, **kwargs):
        return self.action_space.sample()

    def init(self):
        pass
