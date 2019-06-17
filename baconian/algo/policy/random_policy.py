from baconian.core.core import EnvSpec
from baconian.algo.policy.policy import Policy
from typeguard import typechecked
from overrides.overrides import overrides
import numpy as np


class UniformRandomPolicy(Policy):

    @typechecked
    def __init__(self, env_spec: EnvSpec, name: str = 'random_policy'):
        super().__init__(env_spec=env_spec, name=name)

    @overrides
    def forward(self, obs, **kwargs):
        return np.array(self.action_space.sample())

    def save(self, global_step, save_path=None, name=None, **kwargs):
        pass

    def load(self, path_to_model, model_name, global_step=None, **kwargs):
        pass
