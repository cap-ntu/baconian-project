# from baconian.config.global_config import GlobalConfig
# from baconian.config.dict_config import DictConfig
# from baconian.core.pipeline import Pipeline
# from baconian.core.core import Env
# import abc
# from baconian.agent.agent import Agent
# import numpy as np
# from baconian.common.misc import *
# from baconian.common.util.logging import ConsoleLogger
# from baconian.common.util.logging import record_return_decorator
# from baconian.core.status import register_counter_info_to_status_decorator
#
#
# class ModelFreePipeline(Pipeline):
#     STATE_LIST = ['state_not_inited', 'state_inited', 'state_agent_testing', 'state_agent_training', 'state_ended',
#                   'state_corrupted']
#     INIT_STATE = 'state_not_inited'
#
#     required_key_dict = DictConfig.load_json(file_path=GlobalConfig.DEFAULT_MODEL_FREE_PIPELINE_REQUIRED_KEY_LIST)
#
#     def __init__(self, config_or_config_dict: (DictConfig, dict), agent: Agent, env: Env):
#         transitions = []
#         self.agent = agent
#         self.env = env
#         config = construct_dict_config(config_or_config_dict, obj=self)
#         super().__init__(config=config, init_state=self.INIT_STATE, states=self.STATE_LIST, transitions=transitions)
#         self.finite_state_machine.add_transition('initializing', 'state_not_inited', 'state_inited')
#         self.finite_state_machine.add_transition('train',
#                                                  ['state_agent_testing', 'state_inited', 'state_agent_training'],
#                                                  'state_agent_training')
#         self.finite_state_machine.add_transition('test',
#                                                  ['state_agent_training', 'state_inited', 'state_agent_testing'],
#                                                  'state_agent_testing')
#         self.finite_state_machine.add_transition('end',
#                                                  ['state_agent_training', 'state_agent_testing', 'state_inited',
#                                                   'state_not_inited'],
#                                                  'state_ended',
#                                                  conditions='_is_flow_ended')
#         self.status_collector.register_info_key_status(obj=agent, info_key='predict_counter',
#                                                        under_status='TRAIN',
#                                                        return_name='TOTAL_AGENT_TRAIN_SAMPLE_COUNT')
#         self.status_collector.register_info_key_status(obj=agent, info_key='predict_counter',
#                                                        under_status='TEST',
#                                                        return_name='TOTAL_AGENT_TEST_SAMPLE_COUNT')
#         self.status_collector.register_info_key_status(obj=agent,
#                                                        info_key='update_counter',
#                                                        under_status='TRAIN',
#                                                        return_name='TOTAL_AGENT_UPDATE_COUNT')
#
#     def launch(self):
#         assert self.is_state_not_inited()
#         try:
#             self.trigger('initializing')
#             while self.is_state_ended() is False:
#                 self.trigger('train')
#                 self.trigger('test')
#                 self.trigger('end')
#         except GlobalConfig.DEFAULT_CATCHED_EXCEPTION_OR_ERROR_LIST as e:
#             self.to_state_corrupted()
#
#     def on_enter_state_not_inited(self):
#         pass
#
#     def on_exit_state_not_inited(self):
#         pass
#
#     def on_enter_state_inited(self):
#         self.agent.init()
#         self.env.init()
#
#     def on_exit_state_inited(self):
#         ConsoleLogger().print('info', 'model-free pipeline finish inited')
#
#     def on_enter_state_agent_testing(self):
#         ConsoleLogger().print('info', 'model-free pipeline enter testing')
#
#     @record_return_decorator(which_recorder='self')
#     def on_exit_state_agent_testing(self):
#
#         res = self.agent.sample(env=self.agent.env,
#                                 sample_count=self.config('TEST_SAMPLES_COUNT'),
#                                 store_flag=False,
#                                 in_test_flag=True)
#         self.total_test_samples += self.config('TEST_SAMPLES_COUNT')
#         ConsoleLogger().print('info', "Mean reward_func is {}".format(res.get_mean_of(set_name='reward_set')))
#         ConsoleLogger().print('info', 'model-free pipeline exit testing')
#         return dict(average_test_reward=res.get_mean_of(set_name='reward_set'))
#
#     def on_enter_state_agent_training(self):
#         ConsoleLogger().print('info', 'model-free pipeline enter training')
#
#     @record_return_decorator(which_recorder='self')
#     def on_exit_state_agent_training(self):
#         res = self.agent.sample(env=self.agent.env,
#                                 sample_count=self.config('TRAIN_SAMPLES_COUNT'),
#                                 store_flag=True,
#                                 in_test_flag=False)
#         self.total_train_samples += self.config('TRAIN_SAMPLES_COUNT')
#         info = self.agent._update_algo()
#         ConsoleLogger().print('info', "Mean reward_func is {}".format(res.get_mean_of(set_name='reward_set')))
#         ConsoleLogger().print('info', "Train info is {}".format(info))
#         ConsoleLogger().print('info', 'model-free pipeline exit training')
#         return dict(average_train_reward=res.get_mean_of(set_name='reward_set'))
#
#     def on_enter_state_ended(self):
#         ConsoleLogger().print('info', 'model-free pipeline enter ended')
#
#     def on_exit_state_ended(self):
#         ConsoleLogger().print('info', 'model-free pipeline exit ended')
#
#     def on_enter_state_corrupted(self):
#         raise NotImplementedError
#
#     def on_exit_state_corrupted(self):
#         raise NotImplementedError
#
#     @abc.abstractmethod
#     def _is_flow_ended(self):
#         res = self.status_collector()
#         for key in GlobalConfig.DEFAULT_EXPERIMENT_END_POINT:
#             if key in res and GlobalConfig.DEFAULT_EXPERIMENT_END_POINT[key] and res[key] >= \
#                     GlobalConfig.DEFAULT_EXPERIMENT_END_POINT[key]:
#                 ConsoleLogger().print('info',
#                                       'pipeline ended because {}: {} >= end point value {}'.format(key, res[key],
#                                                                                                    GlobalConfig.DEFAULT_EXPERIMENT_END_POINT[
#                                                                                                        key]))
#                 return True
#
#         return False
