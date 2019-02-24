from mobrl.core.pipeline import Pipeline
from mobrl.config.global_config import GlobalConfig
from mobrl.envs.env import Env
from mobrl.config.dict_config import DictConfig
from mobrl.agent.agent import Agent
import numpy as np
from mobrl.common.misc import *


class ModelBasedPipeline(Pipeline):
    """
    This class implement a very naive model based rl pipeline which is:
    train dynamics model -> train agent -> test agent -> train dynamics model ... -> end
    """
    STATE_LIST = ['state_not_inited', 'state_inited', 'state_agent_testing', 'state_agent_training', 'state_ended',
                  'state_dynamics_testing', 'state_dynamics_training']
    INIT_STATE = 'state_not_inited'
    required_key_list = DictConfig.load_json(file_path=GlobalConfig.DEFAULT_MODEL_BASED_PIPELINE_REQUIRED_KEY_LIST)

    def __init__(self, config_or_config_dict: (DictConfig, dict), agent: Agent, env: Env):
        transitions = []
        self.agent = agent
        self.env = env
        config = construct_dict_config(config_or_config_dict, obj=self)
        super().__init__(config=config, init_state=self.INIT_STATE, states=self.STATE_LIST, transitions=transitions)

        # todo move the hard code here
        self.finite_state_machine.add_transition('init', 'state_not_inited', 'state_inited')
        self.finite_state_machine.add_transition('train agent',
                                                 ['state_agent_testing', 'state_inited', 'state_agent_training',
                                                  'state_dynamics_testing', 'state_dynamics_training'],
                                                 'state_agent_training')
        self.finite_state_machine.add_transition('test agent',
                                                 ['state_agent_training', 'state_inited', 'state_agent_testing',
                                                  'state_dynamics_testing', 'state_dynamics_training'],
                                                 'state_agent_testing')

        self.finite_state_machine.add_transition('train dynamics',
                                                 ['state_agent_testing', 'state_inited', 'state_agent_training',
                                                  'state_dynamics_testing', 'state_dynamics_training'],
                                                 'state_dynamics_training')
        self.finite_state_machine.add_transition('test dynamics',
                                                 ['state_agent_training', 'state_inited', 'state_agent_testing',
                                                  'state_dynamics_testing', 'state_dynamics_training'],
                                                 'state_dynamics_testing')

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
                self.trigger('train dynamics')
                self.trigger('test dynamics')
                self.trigger('train agent')
                self.trigger('test agent')
                self.trigger('end')
        except GlobalConfig.DEFAULT_CATCHED_EXCEPTION_OR_ERROR_LIST as e:
            self.to_state_corrupted()

    def on_enter_state_inited(self):
        self.env.init()
        self.agent.init()

    def on_exit_state_inited(self):
        print('model-based pipeline finish inited')

    def on_enter_state_agent_testing(self):
        pass

    def on_exit_state_agent_testing(self):
        res = self.agent.sample(env=self.agent.env,
                                sample_count=self.config('TEST_SAMPLES_COUNT'),
                                store_flag=False,
                                in_test_flag=True)
        self.total_test_samples += self.config('TEST_SAMPLES_COUNT')
        print("Mean reward_func is {}".format(np.mean(res.reward_set)))

        print('model-based pipeline exit testing')

    def on_enter_state_agent_training(self):
        pass

    def on_exit_state_agent_training(self):
        res = self.agent.sample(env=self.agent.env,
                                sample_count=self.config('TRAIN_SAMPLES_COUNT'),
                                store_flag=True,
                                in_test_flag=False)
        self.total_train_samples += self.config('TRAIN_SAMPLES_COUNT')
        info = self.agent.update(state=self.state)
        print("Mean reward_func is {}".format(np.mean(res.reward_set)))
        print("Train info is {}".format(info))
        print('model-based pipeline exit training')

    def on_enter_state_dynamics_testing(self):
        pass

    def on_exit_state_dynamics_testing(self):
        pass

    def on_enter_state_dynamics_training(self):
        pass

    def on_exit_state_dynamics_training(self):
        res = self.agent.sample(env=self.agent.env,
                                sample_count=self.config('TRAIN_SAMPLES_COUNT'),
                                store_flag=True,
                                in_test_flag=False)
        self.total_train_samples += self.config('TRAIN_SAMPLES_COUNT')
        info = self.agent.update(batch_data=res, state=self.state)
        print("Mean reward_func is {}".format(np.mean(res.reward_set)))
        print("Train info is {}".format(info))
        print('model-based pipeline exit training')

    def on_enter_state_ended(self):
        print('model-based pipeline enter ended')

    def on_exit_state_ended(self):
        print('model-based pipeline exit ended')

    def on_enter_state_corrupted(self):
        pass

    def on_exit_state_corrupted(self):
        pass

    def _is_flow_ended(self):
        return self.total_train_samples >= self.config("TOTAL_SAMPLES_COUNT")

    def on_enter_state_not_inited(self):
        pass

    def on_exit_state_not_inited(self):
        pass
