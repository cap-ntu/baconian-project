from src.config.global_config import GlobalConfig
from src.config.dict_config import DictConfig
from src.core.pipeline import Pipeline
from src.envs.env import Env
import abc
from src.agent.agent import Agent
import numpy as np
from src.common.misc import *


class ModelFreePipeline(Pipeline):
    STATE_LIST = ['state_not_inited', 'state_inited', 'state_agent_testing', 'state_agent_training', 'state_ended',
                  'state_corrupted']
    INIT_STATE = 'state_not_inited'

    required_key_list = DictConfig.load_json(file_path=GlobalConfig.DEFAULT_MODEL_FREE_PIPELINE_REQUIRED_KEY_LIST)

    def __init__(self, config_or_config_dict: (DictConfig, dict), agent: Agent, env: Env):
        transitions = []
        self.agent = agent
        self.env = env
        config = construct_dict_config(config_or_config_dict, obj=self)
        super().__init__(config=config, init_state=self.INIT_STATE, states=self.STATE_LIST, transitions=transitions)
        # todo move the hard code here
        self.finite_state_machine.add_transition('init', 'state_not_inited', 'state_inited')
        self.finite_state_machine.add_transition('train',
                                                 ['state_agent_testing', 'state_inited', 'state_agent_training'],
                                                 'state_agent_training')
        self.finite_state_machine.add_transition('test',
                                                 ['state_agent_training', 'state_inited', 'state_agent_testing'],
                                                 'state_agent_testing')
        self.finite_state_machine.add_transition('end',
                                                 ['state_agent_training', 'state_agent_testing', 'state_inited',
                                                  'state_not_inited'],
                                                 'state_ended',
                                                 conditions='_is_flow_ended')

    def launch(self):
        assert self.is_state_not_inited()
        try:
            self.trigger('init')
            while self.is_state_ended() is False:
                self.trigger('train')
                self.trigger('test')
                self.trigger('end')
        except GlobalConfig.DEFAULT_CATCHED_EXCEPTION_OR_ERROR_LIST as e:
            self.to_state_corrupted()

    def on_enter_state_not_inited(self):
        pass

    def on_exit_state_not_inited(self):
        pass

    def on_enter_state_inited(self):
        self.agent.init()
        self.env.init()

    def on_exit_state_inited(self):
        print('model-free pipeline finish inited')

    def on_enter_state_agent_testing(self):
        print('model-free pipeline enter testing')

    def on_exit_state_agent_testing(self):

        res = self.agent.sample(env=self.agent.env,
                                sample_count=self.config('TEST_SAMPLES_COUNT'),
                                store_flag=False,
                                in_test_flag=True)
        self.total_test_samples += self.config('TEST_SAMPLES_COUNT')
        print("Mean reward_func is {}".format(np.mean(res.reward_set)))

        print('model-free pipeline exit testing')

    def on_enter_state_agent_training(self):
        print('model-free pipeline enter training')

    def on_exit_state_agent_training(self):
        res = self.agent.sample(env=self.agent.env,
                                sample_count=self.config('TRAIN_SAMPLES_COUNT'),
                                store_flag=True,
                                in_test_flag=False)
        self.total_train_samples += self.config('TRAIN_SAMPLES_COUNT')
        info = self.agent.update()
        print("Mean reward_func is {}".format(np.mean(res.reward_set)))
        print("Train info is {}".format(info))
        print('model-free pipeline exit training')

    def on_enter_state_ended(self):
        print('model-free pipeline enter ended')

    def on_exit_state_ended(self):
        print('model-free pipeline exit ended')

    def on_enter_state_corrupted(self):
        raise NotImplementedError

    def on_exit_state_corrupted(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _is_flow_ended(self):
        return self.total_train_samples >= self.config("TOTAL_SAMPLES_COUNT")
        # raise NotImplementedError
