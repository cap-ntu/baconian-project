"""
iLQR benchmark on Pendulum
"""

from baconian.benchmark.iLQR_benchmark.pendulum_conf import *
from baconian.core.flow.dyna_flow import DynaFlow
from baconian.envs.envs_reward_func import REWARD_FUNC_DICT
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.core.agent import Agent
from baconian.core.experiment import Experiment
from baconian.algo.dynamics.mlp_dynamics_model import ContinuousMLPGlobalDynamicsModel
from baconian.config.global_config import GlobalConfig
from baconian.core.status import get_global_status_collect
from baconian.algo.policy.ilqr_policy import iLQRPolicy, iLQRAlogWrapper
from baconian.algo.dynamics.reward_func.reward_func import RewardFuncCostWrapper
from baconian.algo.dynamics.dynamics_model import DynamicsEnvWrapper
from baconian.algo.dynamics.terminal_func.terminal_func import FixedEpisodeLengthTerminalFunc


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
    dyna_env = DynamicsEnvWrapper(mlp_dyna)
    dyna_env.set_terminal_reward_func(
        terminal_func=FixedEpisodeLengthTerminalFunc(max_step_length=env.unwrapped._max_episode_steps,
                                                     step_count_fn=dyna_env.total_step_count_fn),
        reward_func=REWARD_FUNC_DICT['Pendulum-v0']())

    policy = iLQRPolicy(env_spec=env_spec,
                        **exp_config['ILQR'],
                        dynamics=dyna_env,
                        cost_fn=RewardFuncCostWrapper(reward_func=REWARD_FUNC_DICT['Pendulum-v0']()))

    algo = iLQRAlogWrapper(policy=policy,
                           env_spec=env_spec,
                           dynamics_env=dyna_env)

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
                               'kwargs': dict(state='state_dynamics_training')},
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
