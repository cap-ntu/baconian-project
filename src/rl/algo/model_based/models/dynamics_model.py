from src.core.basic import Basic
from src.envs.env_spec import EnvSpec
import numpy as np
import abc
from src.core.parameters import Parameters


class DynamicsModel(Basic):
    # todo how to define the APIs, especially for training utility
    def __init__(self, env_spec: EnvSpec, parameters: Parameters, init_state=None):
        super().__init__()
        self.env_space = env_spec
        self.state = init_state
        self.parameters = parameters

    def init(self):
        raise NotImplementedError

    def step(self, action: np.ndarray, state=None, **kwargs_for_transit):
        # todo is this function shall be abstract or implemented in this way?
        state = state if state is not None else self.state
        assert self.env_space.action_space.contains(action)
        assert self.env_space.obs_space.contains(state)
        # todo need flat to convert the action
        new_state = self._state_transit(state=state, action=EnvSpec.flat(self.env_space.action_space, action),
                                        **kwargs_for_transit)
        assert self.env_space.obs_space.contains(new_state)
        return new_state

    @abc.abstractmethod
    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def train(self, batch_data, **kwargs):
        raise NotImplementedError

    def copy(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        return True
