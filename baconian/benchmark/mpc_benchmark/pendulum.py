"""
MPC benchmark on Pendulum
"""

from baconian.benchmark.mpc_benchmark.pendulum_conf import *
from baconian.algo.dynamics.terminal_func.terminal_func import FixedEpisodeLengthTerminalFunc
from baconian.core.flow.dyna_flow import DynaFlow
from baconian.envs.envs_reward_func import REWARD_FUNC_DICT
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.core.agent import Agent
from baconian.core.experiment import Experiment
from baconian.algo.mpc import ModelPredictiveControl
from baconian.algo.policy import UniformRandomPolicy
from baconian.algo.dynamics.mlp_dynamics_model import ContinuousMLPGlobalDynamicsModel
from baconian.config.global_config import GlobalConfig
from baconian.core.status import get_global_status_collect


def pendulum_task_fn():
    exp_config = PENDULUM_BENCHMARK_CONFIG_DICT
    GlobalConfig().set('DEFAULT_EXPERIMENT_END_POINT',
                       exp_config['DEFAULT_EXPERIMENT_END_POINT'])

    env = make('Pendulum-v0')
    name = 'benchmark'
    env_spec = EnvSpec(obs_space=env.observation_space,
                       action_space=env.action_space)

    mlp_dyna = ContinuousMLPGlobalDynamicsModel(
        env_spec=env_spec,
        name_scope=name + '_mlp_dyna',
        name=name + '_mlp_dyna',
        **exp_config['DynamicsModel']
    )
    algo = ModelPredictiveControl(
        dynamics_model=mlp_dyna,
        env_spec=env_spec,
        config_or_config_dict=exp_config['MPC'],
        name=name + '_mpc',
        policy=UniformRandomPolicy(env_spec=env_spec, name='uni_policy')
    )
    algo.set_terminal_reward_function_for_dynamics_env(reward_func=REWARD_FUNC_DICT['Pendulum-v0'](),
                                                       terminal_func=FixedEpisodeLengthTerminalFunc(
                                                           max_step_length=env.unwrapped._max_episode_steps,
                                                           step_count_fn=algo.dynamics_env.total_step_count_fn), )
    agent = Agent(env=env, env_spec=env_spec,
                  algo=algo,
                  exploration_strategy=None,
                  noise_adder=None,
                  name=name + '_agent')

    flow = DynaFlow(
        train_sample_count_func=lambda: get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
        config_or_config_dict=exp_config['DynaFlow'],
        func_dict={
            'train_dynamics': {'func': agent.train,
                               'args': list(),
                               'kwargs': dict()},
            'train_algo': None,
            'test_algo': {'func': agent.test,
                          'args': list(),
                          'kwargs': dict(sample_count=1)},
            'test_dynamics': {'func': agent.algo.test_dynamics,
                              'args': list(),
                              'kwargs': dict(sample_count=100, env=env)},
            'sample_from_real_env': {'func': agent.sample,
                                     'args': list(),
                                     'kwargs': dict(sample_count=10,
                                                    env=agent.env,
                                                    in_which_status='TRAIN',
                                                    store_flag=True)},
            'sample_from_dynamics_env': None,
            'train_algo_from_synthesized_data': None
        }
    )

    experiment = Experiment(
        tuner=None,
        env=env,
        agent=agent,
        flow=flow,
        name=name
    )
    experiment.run()
