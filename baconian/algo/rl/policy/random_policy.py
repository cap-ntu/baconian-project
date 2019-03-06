from baconian.core.core import EnvSpec
from baconian.algo.rl.policy.policy import Policy
from typeguard import typechecked
from overrides.overrides import overrides


class UniformRandomPolicy(Policy):

    @typechecked
    def __init__(self, name: str, env_spec: EnvSpec):
        super().__init__(env_spec=env_spec, name=name)

    @overrides
    def forward(self, obs, **kwargs):
        return self.action_space.sample()

    def init(self):
        pass

    def save(self, global_step, save_path=None, name=None, **kwargs):
        pass

    def load(self, path_to_model, model_name, global_step=None, **kwargs):
        pass
