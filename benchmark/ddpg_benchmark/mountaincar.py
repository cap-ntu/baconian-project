"""
DDPG bechmark on Mountain Car
"""
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.algo.ddpg import DDPG
from baconian.algo.policy import DeterministicMLPPolicy
from baconian.core.agent import Agent
from baconian.core.experiment import Experiment
from baconian.core.flow.train_test_flow import TrainTestFlow
from baconian.config.global_config import GlobalConfig
from benchmark.ddpg_benchmark.mountain_car_continuous_conf import *
from baconian.core.status import get_global_status_collect
from baconian.common.noise import *
from baconian.common.schedules import *


def mountiancar_task_fn():
    exp_config = MOUNTAIN_CAR_CONTINUOUS_BENCHMARK_CONFIG_DICT

    GlobalConfig().set('DEFAULT_EXPERIMENT_END_POINT',
                       exp_config['DEFAULT_EXPERIMENT_END_POINT'])

    env = make('MountainCarContinuous-v0')
    name = 'benchmark'
    env_spec = EnvSpec(obs_space=env.observation_space,
                       action_space=env.action_space)

    mlp_q = MLPQValueFunction(env_spec=env_spec,
                              name_scope=name + '_mlp_q',
                              name=name + '_mlp_q',
                              **exp_config['MLPQValueFunction'])
    policy = DeterministicMLPPolicy(env_spec=env_spec,
                                    name_scope=name + '_mlp_policy',
                                    name=name + '_mlp_policy',
                                    output_low=env_spec.action_space.low,
                                    output_high=env_spec.action_space.high,
                                    **exp_config['DeterministicMLPPolicy'],
                                    reuse=False)

    ddpg = DDPG(
        env_spec=env_spec,
        policy=policy,
        value_func=mlp_q,
        name=name + '_ddpg',
        **exp_config['DDPG']
    )
    agent = Agent(env=env, env_spec=env_spec,
                  algo=ddpg,
                  exploration_strategy=None,
                  noise_adder=AgentActionNoiseWrapper(noise=UniformNoise(0.4),
                                                      # noise_weight_scheduler=
                                                      # noise_weight_scheduler=ConstantScheduler(1),
                                                      # action_weight_scheduler=ConstantScheduler(0), ),
                                                      noise_weight_scheduler=LinearScheduler(initial_p=1.0,
                                                                                             final_p=0.0,
                                                                                             schedule_timesteps= 100000,
                                                                                             t_fn=lambda: get_global_status_collect()(
                                                                                                'TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                                                                                             ),
                                                      action_weight_scheduler=LinearScheduler(initial_p=0.0,
                                                                                              final_p=1.0,
                                                                                              schedule_timesteps=100000,
                                                                                              t_fn=lambda: get_global_status_collect()(
                                                                                                 'TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                                                                                              ), ),
                  name=name + '_agent')

    flow = TrainTestFlow(train_sample_count_func=lambda: get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                         config_or_config_dict=exp_config['TrainTestFlow']['config_or_config_dict'],
                         func_dict={
                             'test': {'func': agent.test,
                                      'args': list(),
                                      'kwargs': dict(sample_count=exp_config['TrainTestFlow']['TEST_SAMPLES_COUNT'],
                                                     sample_trajectory_flag=True),
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

if __name__ == "__main__":
    from baconian.core.experiment_runner import *

    GlobalConfig().set('DEFAULT_LOG_PATH', './mountain_log_path')
    single_exp_runner(mountiancar_task_fn, del_if_log_path_existed=True)
