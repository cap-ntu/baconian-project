from mbrl.core.basic import Basic
from mbrl.envs.env_spec import EnvSpec
import numpy as np
import abc
from mbrl.core.parameters import Parameters


class DynamicsModel(Basic):
    def __init__(self, env_spec: EnvSpec, parameters: Parameters = None, init_state=None):
        super().__init__()
        self.env_space = env_spec
        self.state = init_state
        self.parameters = parameters

    def init(self):
        raise NotImplementedError

    def step(self, action: np.ndarray, state=None, **kwargs_for_transit):
        state = state if state is not None else self.state
        assert self.env_space.action_space.contains(action)
        assert self.env_space.obs_space.contains(state)
        new_state = self._state_transit(state=state, action=self.env_space.flat_action(action),
                                        **kwargs_for_transit)
        assert self.env_space.obs_space.contains(new_state)
        self.state = new_state
        return new_state

    @abc.abstractmethod
    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def copy(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        return True
