from baconian.core.core import Env, Basic
import gym


class MultiAgentEnv(gym.Env, Basic):
    key_list = []
    STATUS_LIST = ['JUST_RESET', 'JUST_INITED', 'TRAIN', 'TEST', 'NOT_INITED']
    INIT_STATUS = 'NOT_INITED'
    """
    Abstract class for multi agent environment
    """

    def __init__(self, agent_count: int, name: str, status=None):
        super().__init__(name, status)
        self.agent_count = agent_count

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
