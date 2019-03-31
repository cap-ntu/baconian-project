"""
DDPG bechmark
"""
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from baconian.algo.rl.model_free.ddpg import DDPG
from baconian.algo.rl.policy.deterministic_mlp import DeterministicMLPPolicy
from baconian.core.agent import Agent
from baconian.algo.rl.misc.epsilon_greedy import EpsilonGreedy
from baconian.core.experiment import Experiment
from baconian.core.pipelines.train_test_flow import TrainTestFlow
from baconian.config.global_config import GlobalConfig
from benchmark.ddpg_bechmark.mountain_car_continuous_conf import *
from benchmark.ddpg_bechmark.pendulum_conf import *
from baconian.common.schedules import LinearSchedule
from baconian.core.status import get_global_status_collect


def task_fn(env_id, exp_config):
    env = make(env_id)
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
                  exploration_strategy=EpsilonGreedy(action_space=env_spec.action_space,
                                                     init_random_prob=1.0,
                                                     prob_scheduler=LinearSchedule(
                                                         t_fn=lambda: get_global_status_collect()(
                                                             'TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                                                         **exp_config['EpsilonGreedy'])
                                                     ),
                  name=name + '_agent',
                  **exp_config['Agent'])
    flow = TrainTestFlow(train_sample_count_func=lambda: get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                         config_or_config_dict={
                             "TEST_EVERY_SAMPLE_COUNT": 10,
                             "TRAIN_EVERY_SAMPLE_COUNT": 10,
                             "START_TRAIN_AFTER_SAMPLE_COUNT": 5,
                             "START_TEST_AFTER_SAMPLE_COUNT": 5,
                         },
                         func_dict={
                             'test': {'func': agent.test,
                                      'args': list(),
                                      'kwargs': dict(sample_count=10),
                                      },
                             'train': {'func': agent.train,
                                       'args': list(),
                                       'kwargs': dict(),
                                       },
                             'sample': {'func': agent.sample,
                                        'args': list(),
                                        'kwargs': dict(sample_count=100,
                                                       env=agent.env,
                                                       in_test_flag=False,
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


if __name__ == '__main__':
    import os

    CURRNET_PATH = os.path.dirname(os.path.realpath(__file__))

    ddpg_benchmark_conf = {
        'Pendulum-v0': PENDULUM_BENCHMARK_CONFIG_DICT,
        'MountainCarContinuous-v0': MOUNTAIN_CAR_CONTINUOUS_BENCHMARK_CONFIG_DICT
    }

    env_id = 'MountainCarContinuous-v0'
    assert ddpg_benchmark_conf[env_id]['env_id'] == env_id

    from baconian.core.experiment_runner import single_exp_runner

    GlobalConfig.set('DEFAULT_LOG_PATH', os.path.join(CURRNET_PATH, os.pardir, 'benchmark_log', env_id))
    single_exp_runner(task_fn, env_id=env_id, exp_config=ddpg_benchmark_conf[env_id])
