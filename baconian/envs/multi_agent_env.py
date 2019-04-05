from baconian.core.core import Env, Basic
import gym
from baconian.core.agent import Agent
from typeguard import typechecked
from baconian.common.error import *


class MultiAgentEnv(gym.Env, Basic):
    key_list = []
    STATUS_LIST = ['JUST_RESET', 'JUST_INITED', 'TRAIN', 'TEST', 'NOT_INITED']
    INIT_STATUS = 'NOT_INITED'
    """
    Abstract class for multi agent environment
    """

    def __init__(self, name: str, status=None):
        super().__init__(name, status)
        self._registered_agent_list = []
        self._registered_agent_name_list = []

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

    @typechecked
    def register_agent(self, agent: Agent):
        if agent.name in self._registered_agent_name_list:
            raise DuplicatedRegisteredError('Agent {} already registered'.format(agent.name))
        self._registered_agent_list.append(agent)
        self._registered_agent_name_list.append(agent.name)
