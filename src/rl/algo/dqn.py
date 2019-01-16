# Date: 11/16/18
# Author: Luke
# Project: ModelBasedRLFramework
import os
from src.rl.algo.algo import Algo
from src.core.config import Config
from overrides.overrides import overrides
from typeguard import typechecked
from src.rl.value_func import *
from src.envs.env_spec import EnvSpec
from src.rl.algo.replay_buffer.replay_buffer import UniformRandomReplayBuffer, BaseReplayBuffer
from src.util.required_keys import SRC_UTIL_REQUIRED_KEYS


class DQN(Algo):
    required_key_list = Config.load_json(file_path=os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                                'dqn.json'))

    def __init__(self,
                 env_spec: EnvSpec,
                 value_func: ValueFunction,
                 replay_buffer_size=int(1e5),
                 replay_buffer=None):
        super().__init__(env_spec=env_spec)
        self.q_value_func = value_func
        if replay_buffer:
            assert issubclass(replay_buffer, BaseReplayBuffer)
            self.replay_buffer = replay_buffer
        else:
            self.replay_buffer = UniformRandomReplayBuffer(limit=replay_buffer_size,
                                                           action_shape=env_spec.flat_action_dim,
                                                           observation_shape=env_spec.flat_obs_dim)

    def init(self):
        self.q_value_func.init()

    @typechecked
    def train(self, batch_data: dict):
        pass

    def test(self, *arg, **kwargs):
        pass

    def _set_up_loss(self):

        pass
