"""
A simple example to show how to build up an experiment with ddpg training and testing on MountainCarContinuous-v0
"""
from baconian.core.core import Basic, EnvSpec
from baconian.envs.gym_env import make
from baconian.agent.agent import Agent
from baconian.algo.rl.misc.epsilon_greedy import EpsilonGreedy
from baconian.core.experiment import Experiment
from baconian.core.pipelines.train_test_flow import TrainTestFlow
from baconian.algo.rl.model_based.mpc import ModelPredictiveControl
from baconian.algo.rl.model_based.misc.terminal_func.terminal_func import RandomTerminalFunc
from baconian.algo.rl.model_based.misc.reward_func.reward_func import RandomRewardFunc
from baconian.algo.rl.policy.random_policy import UniformRandomPolicy
from baconian.algo.rl.model_based.models.mlp_dynamics_model import ContinuousMLPGlobalDynamicsModel


def task_fn():
    env = make('Swimmer-v1')
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
        reward_func=RandomRewardFunc(name='reward_func'),
        terminal_func=RandomTerminalFunc(name='random_terminal'),
        policy=UniformRandomPolicy(env_spec=env_spec, name='uni_policy')
    )
    agent = Agent(env=env, env_spec=env_spec,
                  algo=algo,
                  config_or_config_dict={
                      "TEST_SAMPLES_COUNT": 100,
                      "TRAIN_SAMPLES_COUNT": 100,
                      "TOTAL_SAMPLES_COUNT": 500
                  },
                  name=name + '_agent',
                  exploration_strategy=EpsilonGreedy(action_space=env_spec.action_space,
                                                     init_random_prob=0.5))
    experiment = Experiment(
        tuner=None,
        env=env,
        agent=agent,
        flow=TrainTestFlow(),
        name=name
    )
    experiment.run()


if __name__ == '__main__':
    from baconian.core.experiment_runner import single_exp_runner

    single_exp_runner(task_fn, gpu_id=3)
