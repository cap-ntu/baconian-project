"""
A simple example to show how to build up an experiment with ddpg training and testing on MountainCarContinuous-v0
"""
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.core.agent import Agent
from baconian.algo.rl.misc.epsilon_greedy import EpsilonGreedy
from baconian.core.experiment import Experiment
from baconian.core.flow.train_test_flow import TrainTestFlow
from baconian.algo.rl.model_based.mpc import ModelPredictiveControl
from baconian.algo.dynamics.terminal_func.terminal_func import RandomTerminalFunc
from baconian.algo.dynamics.reward_func.reward_func import RandomRewardFunc
from baconian.algo.rl.policy.random_policy import UniformRandomPolicy
from baconian.algo.dynamics.mlp_dynamics_model import ContinuousMLPGlobalDynamicsModel
from baconian.config.global_config import GlobalConfig
from baconian.core.status import get_global_status_collect


def task_fn():
    env = make('Pendulum-v0')
    name = 'demo_exp'
    env_spec = EnvSpec(obs_space=env.observation_space,
                       action_space=env.action_space)

    mlp_dyna = ContinuousMLPGlobalDynamicsModel(
        env_spec=env_spec,
        name_scope=name + '_mlp_dyna',
        name=name + '_mlp_dyna',
        output_low=env_spec.obs_space.low,
        output_high=env_spec.obs_space.high,
        l1_norm_scale=1.0,
        l2_norm_scale=1.0,
        learning_rate=0.01,
        mlp_config=[
            {
                "ACT": "RELU",
                "B_INIT_VALUE": 0.0,
                "NAME": "1",
                "N_UNITS": 16,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "LINEAR",
                "B_INIT_VALUE": 0.0,
                "NAME": "OUPTUT",
                "N_UNITS": env_spec.flat_obs_dim,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            }
        ])
    algo = ModelPredictiveControl(
        dynamics_model=mlp_dyna,
        env_spec=env_spec,
        config_or_config_dict=dict(
            SAMPLED_HORIZON=2,
            SAMPLED_PATH_NUM=5,
            dynamics_model_train_iter=10
        ),
        name=name + '_mpc',

        policy=UniformRandomPolicy(env_spec=env_spec, name='uni_policy')
    )
    algo.set_terminal_reward_function_for_dynamics_env(reward_func=RandomRewardFunc(name='reward_func'),
                                                       terminal_func=RandomTerminalFunc(name='random_terminal'), )
    agent = Agent(env=env, env_spec=env_spec,
                  algo=algo,
                  name=name + '_agent',
                  exploration_strategy=EpsilonGreedy(action_space=env_spec.action_space,
                                                     init_random_prob=0.5))
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


from baconian.core.experiment_runner import single_exp_runner

GlobalConfig().set('DEFAULT_LOG_PATH', './log_path')
single_exp_runner(task_fn, del_if_log_path_existed=True)
