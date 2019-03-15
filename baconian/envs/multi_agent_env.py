from baconian.core.core import Env, Basic
import gym


class MultiAgentEnv(gym.Env, Basic):
    """
    Abstract class for multi agent environment
    """
    key_list = []
    STATUS_LIST = ['JUST_RESET', 'JUST_INITED', 'TRAIN', 'TEST', 'NOT_INITED']
    INIT_STATUS = 'NOT_INITED'

    def step(self, action):
        pass

    def reset(self):
        pass

    def close(self):
        super().close()

    def seed(self, seed=None):
        return super().seed(seed)

    def get_state(self):
        pass
