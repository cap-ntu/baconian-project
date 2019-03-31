from baconian.core.core import EnvSpec
from baconian.algo.dynamics.dynamics_model import GlobalDynamicsModel

from baconian.core.parameters import Parameters
import numpy as np


class UniformRandomDynamicsModel(GlobalDynamicsModel):
    """
    A dynamics that uniformly return the new state (sample by env_spec.obs_space.sample()),
    can be used for debugging.
    """

    def __init__(self, env_spec: EnvSpec, parameters: Parameters = None, init_state=None, name='dynamics_model'):
        super().__init__(env_spec, parameters, init_state, name)

    def init(self):
        super().init()

    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        return self.env_spec.obs_space.sample()

    def make_copy(self):
        return UniformRandomDynamicsModel(env_spec=self.env_spec)
