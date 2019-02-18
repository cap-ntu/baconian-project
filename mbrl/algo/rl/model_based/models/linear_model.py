import numpy as np
from mbrl.algo.rl.model_based.models.dynamics_model import DynamicsModel
from mbrl.core.parameters import Parameters
from mbrl.envs.env_spec import EnvSpec


class LinearGlobalDynamicsModel(DynamicsModel):
    def __init__(self, env_spec: EnvSpec, parameters: Parameters = None, init_state=None):
        super().__init__(env_spec, parameters, init_state)

    def init(self):
        pass

    def step(self, action: np.ndarray, state=None, **kwargs_for_transit):
        return super().step(action, state, **kwargs_for_transit)

    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        pass

    def copy_from(self, obj) -> bool:
        return super().copy_from(obj)

    def get_status(self):
        return super().get_status()
