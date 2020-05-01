"""
An example showing the Model-ensemble method in
Kurutach, Thanard, et al. "Model-ensemble trust-region policy optimization." arXiv preprint arXiv:1802.10592 (2018).

Here we use Model-ensemble with DDPG instead of TRPO on Pendulum-v1, also resue the Dyna flow to show the flexibility of Baconian modules.

"""

from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.algo.ddpg import DDPG
from baconian.algo.policy import DeterministicMLPPolicy
from baconian.core.agent import Agent
from baconian.algo.misc import EpsilonGreedy
from baconian.core.experiment import Experiment
from baconian.config.global_config import GlobalConfig
from baconian.core.status import get_global_status_collect
from baconian.common.schedules import PeriodicalEventSchedule
from baconian.algo.dynamics.mlp_dynamics_model import ContinuousMLPGlobalDynamicsModel
from baconian.algo.model_ensemble import ModelEnsembleAlgo
from baconian.envs.envs_reward_func import PendulumRewardFunc
from baconian.algo.dynamics.terminal_func.terminal_func import FixedEpisodeLengthTerminalFunc
from baconian.core.flow.dyna_flow import create_dyna_flow
from baconian.common.data_pre_processing import RunningStandardScaler
from baconian.core.ensemble import ModelEnsemble


def task_fn():
    env = make('Pendulum-v0')
    name = 'demo_exp'
    env_spec = EnvSpec(obs_space=env.observation_space,
                       action_space=env.action_space)

    mlp_q = MLPQValueFunction(env_spec=env_spec,
                              name_scope=name + '_mlp_q',
                              name=name + '_mlp_q',
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
                                      "N_UNITS": 1,
                                      "TYPE": "DENSE",
                                      "W_NORMAL_STDDEV": 0.03
                                  }
                              ])
    policy = DeterministicMLPPolicy(env_spec=env_spec,
                                    name_scope=name + '_mlp_policy',
                                    name=name + '_mlp_policy',
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
                                            "N_UNITS": env_spec.flat_action_dim,
                                            "TYPE": "DENSE",
                                            "W_NORMAL_STDDEV": 0.03
                                        }
                                    ],
                                    reuse=False)

    ddpg = DDPG(
        env_spec=env_spec,
        config_or_config_dict={
            "REPLAY_BUFFER_SIZE": 10000,
            "GAMMA": 0.999,
            "CRITIC_LEARNING_RATE": 0.001,
            "ACTOR_LEARNING_RATE": 0.001,
            "DECAY": 0.5,
            "BATCH_SIZE": 50,
            "TRAIN_ITERATION": 1,
            "critic_clip_norm": 0.1,
            "actor_clip_norm": 0.1,
        },
        value_func=mlp_q,
        policy=policy,
        name=name + '_ddpg',
        replay_buffer=None
    )
    mlp_dyna_list = []
    for i in range(10):
        mlp_dyna = ContinuousMLPGlobalDynamicsModel(
            env_spec=env_spec,
            name_scope=name + '_mlp_dyna_{}'.format(i),
            name=name + '_mlp_dyna_{}'.format(i),
            learning_rate=0.01,
            state_input_scaler=RunningStandardScaler(dims=env_spec.flat_obs_dim),
            action_input_scaler=RunningStandardScaler(dims=env_spec.flat_action_dim),
            output_delta_state_scaler=RunningStandardScaler(dims=env_spec.flat_obs_dim),
            mlp_config=[
                {
                    "ACT": "RELU",
                    "B_INIT_VALUE": 0.0,
                    "NAME": "1",
                    "L1_NORM": 0.0,
                    "L2_NORM": 0.0,
                    "N_UNITS": 16,
                    "TYPE": "DENSE",
                    "W_NORMAL_STDDEV": 0.03
                },
                {
                    "ACT": "LINEAR",
                    "B_INIT_VALUE": 0.0,
                    "NAME": "OUPTUT",
                    "L1_NORM": 0.0,
                    "L2_NORM": 0.0,
                    "N_UNITS": env_spec.flat_obs_dim,
                    "TYPE": "DENSE",
                    "W_NORMAL_STDDEV": 0.03
                }
            ])
        mlp_dyna_list.append(mlp_dyna)
    dyna_ensemble_model = ModelEnsemble(n_models=10, model=mlp_dyna_list, prediction_type='random', env_spec=env_spec)
    algo = ModelEnsembleAlgo(
        env_spec=env_spec,
        model_free_algo=ddpg,
        dynamics_model=dyna_ensemble_model,
        config_or_config_dict=dict(
            dynamics_model_train_iter=10,
            model_free_algo_train_iter=10,
            validation_trajectory_count=2,
        )
    )
    # For examples only, we use random reward function and terminal function with fixed episode length.
    algo.set_terminal_reward_function_for_dynamics_env(
        terminal_func=FixedEpisodeLengthTerminalFunc(max_step_length=env.unwrapped._max_episode_steps,
                                                     step_count_fn=algo.dynamics_env.total_step_count_fn),
        reward_func=PendulumRewardFunc())
    agent = Agent(env=env, env_spec=env_spec,
                  algo=algo,
                  algo_saving_scheduler=PeriodicalEventSchedule(
                      t_fn=lambda: get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                      trigger_every_step=200,
                      after_t=10),
                  name=name + '_agent',
                  exploration_strategy=EpsilonGreedy(action_space=env_spec.action_space,
                                                     init_random_prob=0.5))

    # we can easily reuse the dyna training flow to implement the Model-ensemble training flow.
    flow = create_dyna_flow(
        train_algo_func=(agent.train, (), dict(state='state_agent_training')),
        train_algo_from_synthesized_data_func=(agent.train, (), dict(state='state_agent_training')),
        train_dynamics_func=(agent.train, (), dict(state='state_dynamics_training')),
        test_algo_func=(agent.test, (), dict(sample_count=10)),
        test_dynamics_func=(agent.algo.test_dynamics, (), dict(sample_count=10, env=env)),
        sample_from_real_env_func=(agent.sample, (), dict(sample_count=10,
                                                          env=agent.env,
                                                          store_flag=True)),
        sample_from_dynamics_env_func=(agent.sample, (), dict(sample_count=10,
                                                              env=agent.algo.dynamics_env,
                                                              store_flag=True)),
        # set this to large enough so agent only use data from dynamics env.
        train_algo_every_real_sample_count_by_data_from_real_env=100,
        train_algo_every_real_sample_count_by_data_from_dynamics_env=100,
        test_algo_every_real_sample_count=100,
        test_dynamics_every_real_sample_count=100,
        train_dynamics_ever_real_sample_count=100,
        start_train_algo_after_sample_count=1,
        start_train_dynamics_after_sample_count=1,
        start_test_algo_after_sample_count=1,
        start_test_dynamics_after_sample_count=1,
        warm_up_dynamics_samples=100
    )

    experiment = Experiment(
        tuner=None,
        env=env,
        agent=agent,
        flow=flow,
        name=name + '_exp'
    )
    experiment.run()


from baconian.core.experiment_runner import *

GlobalConfig().set('DEFAULT_LOG_PATH', './log_path')
GlobalConfig().set('DEFAULT_EXPERIMENT_END_POINT', dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=400))

single_exp_runner(task_fn, del_if_log_path_existed=True)
