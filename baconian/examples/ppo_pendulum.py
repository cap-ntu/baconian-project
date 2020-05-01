# Date: 3/30/19
# Author: Luke
# Project: baconian-internal

"""
A simple example to show how to build up an experiment with ppo training and testing on Pendulum-v0
"""
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.value_func import MLPVValueFunc
from baconian.algo.ppo import PPO
from baconian.algo.policy.normal_distribution_mlp import NormalDistributionMLPPolicy
from baconian.core.agent import Agent
from baconian.algo.misc import EpsilonGreedy
from baconian.core.experiment import Experiment
from baconian.core.flow.train_test_flow import create_train_test_flow
from baconian.config.global_config import GlobalConfig
from baconian.core.status import get_global_status_collect
from baconian.common.schedules import PeriodicalEventSchedule


def task_fn():
    env = make('Pendulum-v0')
    name = 'demo_exp_'
    env_spec = EnvSpec(obs_space=env.observation_space,
                       action_space=env.action_space)

    mlp_v = MLPVValueFunc(env_spec=env_spec,
                          name_scope=name + 'mlp_v',
                          name=name + 'mlp_v',
                          mlp_config=[
                              {
                                  "ACT": "RELU",
                                  "B_INIT_VALUE": 0.0,
                                  "NAME": "1",
                                  "N_UNITS": 16,
                                  "L1_NORM": 0.01,
                                  "L2_NORM": 0.01,
                                  "TYPE": "DENSE",
                                  "W_NORMAL_STDDEV": 0.03
                              },
                              {
                                  "ACT": "LINEAR",
                                  "B_INIT_VALUE": 0.0,
                                  "NAME": "OUPTUT",
                                  "N_UNITS": 1,
                                  "TYPE": "DENSE",
                                  "W_NORMAL_STDDEV": 0.03
                              }
                          ])

    policy = NormalDistributionMLPPolicy(env_spec=env_spec,
                                         name_scope=name + 'mlp_policy',
                                         name=name + 'mlp_policy',
                                         mlp_config=[
                                             {
                                                 "ACT": "RELU",
                                                 "B_INIT_VALUE": 0.0,
                                                 "NAME": "1",
                                                 "L1_NORM": 0.01,
                                                 "L2_NORM": 0.01,
                                                 "N_UNITS": 16,
                                                 "TYPE": "DENSE",
                                                 "W_NORMAL_STDDEV": 0.03
                                             },
                                             {
                                                 "ACT": "LINEAR",
                                                 "B_INIT_VALUE": 0.0,
                                                 "NAME": "OUPTUT",
                                                 "N_UNITS": env_spec.flat_action_dim,
                                                 "TYPE": "DENSE",
                                                 "W_NORMAL_STDDEV": 0.03
                                             }
                                         ],
                                         reuse=False)

    ppo = PPO(
        env_spec=env_spec,
        config_or_config_dict={
            "gamma": 0.995,
            "lam": 0.98,
            "policy_train_iter": 10,
            "value_func_train_iter": 10,
            "clipping_range": None,
            "beta": 1.0,
            "eta": 50,
            "log_var_init": -1.0,
            "kl_target": 0.003,
            "policy_lr": 0.01,
            "value_func_lr": 0.01,
            "value_func_train_batch_size": 10,
            "lr_multiplier": 1.0
        },
        value_func=mlp_v,
        stochastic_policy=policy,
        name=name + 'ppo'
    )
    agent = Agent(env=env, env_spec=env_spec,
                  algo=ppo,
                  algo_saving_scheduler=PeriodicalEventSchedule(
                      t_fn=lambda: get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                      trigger_every_step=20,
                      after_t=10),
                  name=name + 'agent',
                  exploration_strategy=EpsilonGreedy(action_space=env_spec.action_space,
                                                     init_random_prob=0.5))
    flow = create_train_test_flow(
        test_every_sample_count=10,
        train_every_sample_count=10,
        start_test_after_sample_count=5,
        start_train_after_sample_count=5,
        train_func_and_args=(agent.train, (), dict()),
        test_func_and_args=(agent.test, (), dict(sample_count=10)),
        sample_func_and_args=(agent.sample, (), dict(sample_count=100,
                                                     env=agent.env,
                                                     sample_type='trajectory',
                                                     store_flag=True))
    )

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
