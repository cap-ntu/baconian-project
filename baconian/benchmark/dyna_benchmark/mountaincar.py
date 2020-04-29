"""
Dyna benchmark on Pendulum
"""

from baconian.benchmark.dyna_benchmark.mountaincar_conf import MOUNTAIN_CAR_BENCHMARK_CONFIG_DICT as exp_config
from baconian.common.noise import *
from baconian.common.schedules import *
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.algo.ddpg import DDPG
from baconian.algo.policy import DeterministicMLPPolicy
from baconian.core.agent import Agent
from baconian.core.experiment import Experiment
from baconian.config.global_config import GlobalConfig
from baconian.core.status import get_global_status_collect
from baconian.algo.dynamics.mlp_dynamics_model import ContinuousMLPGlobalDynamicsModel
from baconian.algo.dyna import Dyna
from baconian.algo.dynamics.terminal_func.terminal_func import FixedEpisodeLengthTerminalFunc
from baconian.core.flow.dyna_flow import DynaFlow
from baconian.envs.envs_reward_func import REWARD_FUNC_DICT


def pendulum_task_fn():
    GlobalConfig().set('DEFAULT_EXPERIMENT_END_POINT',
                       exp_config['DEFAULT_EXPERIMENT_END_POINT'])

    env = make(exp_config['env_id'])
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

    mlp_dyna = ContinuousMLPGlobalDynamicsModel(
        env_spec=env_spec,
        name_scope=name + '_mlp_dyna',
        name=name + '_mlp_dyna',
        **exp_config['DynamicsModel']
    )
    algo = Dyna(env_spec=env_spec,
                name=name + '_dyna_algo',
                model_free_algo=ddpg,
                dynamics_model=mlp_dyna,
                config_or_config_dict=dict(
                    dynamics_model_train_iter=10,
                    model_free_algo_train_iter=10
                ))
    algo.set_terminal_reward_function_for_dynamics_env(
        terminal_func=FixedEpisodeLengthTerminalFunc(max_step_length=env.unwrapped._max_episode_steps,
                                                     step_count_fn=algo.dynamics_env.total_step_count_fn),
        reward_func=REWARD_FUNC_DICT['Pendulum-v0']())
    agent = Agent(env=env, env_spec=env_spec,
                  algo=algo,
                  exploration_strategy=None,
                  noise_adder=AgentActionNoiseWrapper(noise=OrnsteinUhlenbeckActionNoise(np.zeros(1, ), 0.15),
                                                      noise_weight_scheduler=ConstantScheduler(value=0.3),
                                                      action_weight_scheduler=ConstantScheduler(value=1.0)),
                  name=name + '_agent')

    flow = DynaFlow(
        train_sample_count_func=lambda: get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
        config_or_config_dict=exp_config['DynaFlow'],
        func_dict={
            'train_algo': {'func': agent.train,
                           'args': list(),
                           'kwargs': dict(state='state_agent_training')},
            'train_algo_from_synthesized_data': {'func': agent.train,
                                                 'args': list(),
                                                 # TODO use a decomposed way to represetn the state
                                                 #  e.g., TRAIN:AGENT:CYBER
                                                 'kwargs': dict(state='state_agent_training', train_iter=1)},
            'train_dynamics': {'func': agent.train,
                               'args': list(),
                               'kwargs': dict(state='state_dynamics_training')},
            'test_algo': {'func': agent.test,
                          'args': list(),
                          'kwargs': dict(sample_count=1)},
            'test_dynamics': {'func': agent.algo.test_dynamics,
                              'args': list(),
                              'kwargs': dict(sample_count=10, env=env)},
            'sample_from_real_env': {'func': agent.sample,
                                     'args': list(),
                                     'kwargs': dict(sample_count=20,
                                                    env=agent.env,
                                                    sample_type='transition',
                                                    in_which_status='TRAIN',
                                                    store_flag=True)},
            'sample_from_dynamics_env': {'func': agent.sample,
                                         'args': list(),
                                         'kwargs': dict(sample_count=20,
                                                        sample_type='transition',
                                                        env=agent.algo.dynamics_env,
                                                        in_which_status='TRAIN',
                                                        store_flag=False)}
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
