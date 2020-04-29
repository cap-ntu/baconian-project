"""
DQN benchmark on acrobot
"""
from baconian.benchmark.dqn_benchmark.acrobot_conf import *
from baconian.algo.dqn import DQN
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.core.agent import Agent
from baconian.algo.misc import EpsilonGreedy
from baconian.core.experiment import Experiment
from baconian.core.flow.train_test_flow import TrainTestFlow
from baconian.config.global_config import GlobalConfig
from baconian.common.schedules import LinearScheduler
from baconian.core.status import get_global_status_collect

def acrobot_task_fn():
    exp_config = ACROBOT_BENCHMARK_CONFIG_DICT
    GlobalConfig().set('DEFAULT_EXPERIMENT_END_POINT',
                       exp_config['DEFAULT_EXPERIMENT_END_POINT'])

    env = make('Acrobot-v1')
    name = 'benchmark'
    env_spec = EnvSpec(obs_space=env.observation_space,
                       action_space=env.action_space)

    mlp_q = MLPQValueFunction(env_spec=env_spec,
                              name_scope=name + '_mlp_q',
                              name=name + '_mlp_q',
                              **exp_config['MLPQValueFunction'])
    dqn = DQN(env_spec=env_spec,
              name=name + '_dqn',
              value_func=mlp_q,
              **exp_config['DQN'])
    agent = Agent(env=env, env_spec=env_spec,
                  algo=dqn,
                  name=name + '_agent',
                  exploration_strategy=EpsilonGreedy(action_space=env_spec.action_space,
                                                     prob_scheduler=LinearScheduler(
                                                         t_fn=lambda: get_global_status_collect()(
                                                             'TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                                                         **exp_config['EpsilonGreedy']['LinearScheduler']),
                                                     **exp_config['EpsilonGreedy']['config_or_config_dict']))
    flow = TrainTestFlow(train_sample_count_func=lambda: get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                         config_or_config_dict=exp_config['TrainTestFlow']['config_or_config_dict'],
                         func_dict={
                             'test': {'func': agent.test,
                                      'args': list(),
                                      'kwargs': dict(sample_count=exp_config['TrainTestFlow']['TEST_SAMPLES_COUNT']),
                                      },
                             'train': {'func': agent.train,
                                       'args': list(),
                                       'kwargs': dict(),
                                       },
                             'sample': {'func': agent.sample,
                                        'args': list(),
                                        'kwargs': dict(sample_count=exp_config['TrainTestFlow']['TRAIN_SAMPLES_COUNT'],
                                                       env=agent.env,
                                                       in_which_status='TRAIN',
                                                       store_flag=True),
                                        },
                         })

    experiment = Experiment(
        tuner=None,
        env=env,
        agent=agent,
        flow=flow,
        name=name
    )
    experiment.run()
