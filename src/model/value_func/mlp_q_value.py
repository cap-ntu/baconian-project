# Date: 11/16/18
# Author: Luke
# Project: ModelBasedRLFramework
from src.model.value_func.value_func import TrainableValueFunction
import typeguard as tg
from gym.core import Space
from gym.spaces.multi_discrete import MultiDiscrete
import overrides
import numpy as np


class MlpQValueFunction(TrainableValueFunction):
    """
    Multi Layer Q Value Function, based on Tensorflow, take the state/(state, action) as input,
    return the Q value for all action/ input action.
    """

    @tg.typechecked
    def __init__(self,
                 state_space: Space,
                 action_space: Space,
                 auto_set_up: bool = False,
                 q_for_all_action: bool = False,
                 running_normalization: bool = False):
        super().__init__(auto_set_up)
        self.state_space = state_space
        self.action_space = action_space

    @overrides.overrides
    def copy(self, obj) -> bool:
        return super().copy(obj)

    def forward(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def init_set_up(self, *args, **kwargs) -> bool:
        return super().init_set_up(*args, **kwargs)

    @tg.typechecked
    def _forward_single_action(self,
                               state_input: (np.ndarray, list),
                               action_input: (np.ndarray, list)):
        pass

    @tg.typechecked
    def _forward_all_action(self, state_input: (np.ndarray, list)):
        pass
