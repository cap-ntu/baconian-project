from src.core.global_config import GlobalConfig
from src.core.config import Config
from src.core.pipeline import Pipeline
from src.rl import ModelFreeAlgo
from src.envs.env import Env
import abc
from src.agent.agent import Agent
import numpy as np
import os
from src.util.required_keys import SRC_UTIL_REQUIRED_KEYS


class ModelFreePipeline(Pipeline):
    STATE_LIST = ['state_not_inited', 'state_inited', 'state_algo_testing', 'state_algo_training', 'state_ended',
                  'state_corrupted']
    INITE_STATE = 'state_not_inited'

    required_key_list = Config.load_json(file_path=os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                                'model_free_pipeline.json'))

    def __init__(self, config_or_config_dict: (Config, dict), agent: Agent, env: Env):
        transitions = []
        self.agent = agent
        self.env = env
        if isinstance(config_or_config_dict, dict):
            config = Config(required_key_dict=self.required_key_list,
                            cls_name=type(self).__name__,
                            config_dict=config_or_config_dict)

        super().__init__(config=config, init_state=self.INITE_STATE, states=self.STATE_LIST, transitions=transitions)
        self.finite_state_machine.add_transition('init', 'state_not_inited', 'state_inited')
        self.finite_state_machine.add_transition('train',
                                                 ['state_algo_testing', 'state_inited', 'state_algo_training'],
                                                 'state_algo_training')
        self.finite_state_machine.add_transition('test',
                                                 ['state_algo_training', 'state_inited', 'state_algo_testing'],
                                                 'state_algo_testing')
        self.finite_state_machine.add_transition('end',
                                                 ['state_algo_training', 'state_algo_testing', 'state_inited',
                                                  'state_not_inited'],
                                                 'state_ended',
                                                 conditions='_is_flow_ended')
        self.total_test_samples = 0
        self.total_train_samples = 0
        for state_name in self.STATE_LIST:
            assert hasattr(self, 'on_enter_{}'.format(state_name))
            assert hasattr(self, 'on_exit_{}'.format(state_name))

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
        self.agent.init()
        self.env.init()

    def on_exit_state_not_inited(self):
        print('model-free pipeline finish inited')

    def on_enter_state_inited(self):
        self.agent.init()
        self.env.init()

    def on_exit_state_inited(self):
        print('model-free pipeline finish inited')

    def on_enter_state_algo_testing(self):
        print('model-free pipeline enter testing')

    def on_exit_state_algo_testing(self):

        res = self.agent.sample(env=self.agent.env,
                                sample_count=self.config('TEST_SAMPLES_COUNT'),
                                store_flag=False,
                                in_test_flag=True)
        self.total_test_samples += self.config('TEST_SAMPLES_COUNT')
        print("Mean reward is {}".format(np.mean(res.reward_set)))

        print('model-free pipeline exit testing')

    def on_enter_state_algo_training(self):
        print('model-free pipeline enter training')

    def on_exit_state_algo_training(self):
        res = self.agent.sample(env=self.agent.env,
                                sample_count=self.config('TRAIN_SAMPLES_COUNT'),
                                store_flag=True,
                                in_test_flag=False)
        self.total_train_samples += self.config('TRAIN_SAMPLES_COUNT')
        info = self.agent.update()
        print("Mean reward is {}".format(np.mean(res.reward_set)))
        print("Train info is {}".format(info))
        print('model-free pipeline exit training')

    def on_enter_state_ended(self):
        print('model-free pipeline enter ended')

    def on_exit_state_ended(self):
        print('model-free pipeline exit ended')

    def on_enter_state_corrupted(self):
        pass

    def on_exit_state_corrupted(self):
        pass

    @abc.abstractmethod
    def _is_flow_ended(self):
        return self.total_train_samples >= self.config("TOTAL_SAMPLES_COUNT")
        # raise NotImplementedError
