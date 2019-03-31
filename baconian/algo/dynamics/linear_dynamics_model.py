from baconian.core.core import EnvSpec
from baconian.algo.dynamics.dynamics_model import GlobalDynamicsModel

from baconian.core.parameters import Parameters
import numpy as np
from copy import deepcopy


class LinearDynamicsModel(GlobalDynamicsModel):
    """
    A dynamics that uniformly return the new state (sample by env_spec.obs_space.sample()),
    can be used for debugging.
    """

    def __init__(self, env_spec: EnvSpec, state_transition_matrix: np.array, bias: np.array, init_state=None,
                 name='dynamics_model'):
        parameters = Parameters(parameters=dict(F=state_transition_matrix, f=bias))
        super().__init__(env_spec, parameters, init_state, name)
        assert self.parameters('F').shape == \
               (env_spec.obs_space.flat_dim, env_spec.obs_space.flat_dim + env_spec.action_space.flat_dim)
        assert self.parameters('f').shape[0] == env_spec.obs_space.flat_dim

    def init(self):
        super().init()

    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        new_state = np.dot(self.parameters('F'), np.concatenate((state, action))) + self.parameters('f')
        return self.env_spec.obs_space.clip(new_state)

    def make_copy(self):
        return LinearDynamicsModel(env_spec=self.env_spec,
                                   state_transition_matrix=deepcopy(self.parameters('F')),
                                   bias=deepcopy(self.parameters('f')))

    @property
    def F(self):
        return self.parameters('F')

    @property
    def f(self):
        return self.parameters('f')
