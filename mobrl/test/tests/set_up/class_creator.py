import numpy as np
import unittest
import tensorflow as tf
from mobrl.algo.rl.model_based.models.mlp_dynamics_model import ContinuousMLPGlobalDynamicsModel
from mobrl.algo.placeholder_input import PlaceholderInput
from mobrl.algo.rl.model_free.dqn import DQN
from mobrl.tf.tf_parameters import TensorflowParameters
from mobrl.core.core import Basic, EnvSpec
from mobrl.envs.gym_env import make
from mobrl.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from mobrl.algo.rl.model_free.ddpg import DDPG
from mobrl.algo.rl.policy.deterministic_mlp import DeterministicMLPPolicy
from mobrl.algo.rl.value_func.mlp_v_value import MLPVValueFunc
from mobrl.algo.rl.policy.normal_distribution_mlp import NormalDistributionMLPPolicy
from mobrl.algo.rl.model_free.ppo import PPO
from mobrl.core.parameters import Parameters, DictConfig
from mobrl.algo.rl.model_based.mpc import ModelPredictiveControl
from mobrl.algo.rl.model_based.misc.terminal_func.terminal_func import RandomTerminalFunc
from mobrl.algo.rl.model_based.misc.reward_func.reward_func import RandomRewardFunc
from mobrl.algo.rl.policy.random_policy import UniformRandomPolicy
from mobrl.agent.agent import Agent
from mobrl.algo.rl.misc.exploration_strategy.epsilon_greedy import EpsilonGreedy
from mobrl.core.pipelines.model_free_pipelines import ModelFreePipeline
from mobrl.core.experiment import Experiment


class Foo(Basic):
    def __init__(self):
        super().__init__(name='foo')

    required_key_dict = dict(var1=1, var2=0.1)


class ClassCreatorSetup(unittest.TestCase):
    def create_tf_parameters(self, name='test_tf_param'):
        with tf.variable_scope(name):
            a = tf.get_variable(shape=[3, 4], dtype=tf.float32, name='var_1')
            b = tf.get_variable(shape=[3, 4], dtype=tf.bool, name='var_2')

        conf = DictConfig(required_key_dict=Foo.required_key_dict,
                          config_dict=dict(var1=1, var2=0.01))
        param = TensorflowParameters(tf_var_list=[a, b],
                                     rest_parameters=dict(var3='sss'),
                                     name=name,
                                     source_config=conf,
                                     require_snapshot=True,
                                     to_ph_parameter_dict=dict(var1=tf.placeholder(shape=(), dtype=tf.int32)),
                                     auto_init=False)
        return param, locals()

    def create_dqn(self, env_id='Acrobot-v1', name='dqn'):
        env = make(env_id)
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        mlp_q = MLPQValueFunction(env_spec=env_spec,
                                  name_scope=name + 'mlp_q',
                                  name=name + 'mlp_q',
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
        dqn = DQN(env_spec=env_spec,
                  adaptive_learning_rate=True,
                  config_or_config_dict=dict(REPLAY_BUFFER_SIZE=1000,
                                             GAMMA=0.99,
                                             BATCH_SIZE=10,
                                             Q_NET_L1_NORM_SCALE=0.001,
                                             Q_NET_L2_NORM_SCALE=0.001,
                                             LEARNING_RATE=0.001,
                                             TRAIN_ITERATION=1,
                                             DECAY=0.5),
                  name=name + 'dqn',
                  value_func=mlp_q)
        return dqn, locals()

    def create_ph(self, name):
        with tf.variable_scope(name):
            a = tf.get_variable(shape=[3, 4], dtype=tf.float32, name='var_1')

        conf = DictConfig(required_key_dict=Foo.required_key_dict,
                          config_dict=dict(var1=1, var2=0.01))
        param = TensorflowParameters(tf_var_list=[a],
                                     rest_parameters=dict(var3='sss'),
                                     name=name,
                                     source_config=conf,
                                     require_snapshot=True,
                                     to_ph_parameter_dict=dict(var1=tf.placeholder(shape=(), dtype=tf.int32)),
                                     auto_init=False)
        param.init()
        a = PlaceholderInput(parameters=param, inputs=None)

        return a, locals()

    def create_ddpg(self, env_id='Swimmer-v1', name='ddpg'):
        env = make(env_id)
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        mlp_q = MLPQValueFunction(env_spec=env_spec,
                                  name_scope=name + 'mlp_q',
                                  name=name + 'mlp_q',
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
        self.assertTrue(len(mlp_q.parameters('tf_var_list')) == 4)
        policy = DeterministicMLPPolicy(env_spec=env_spec,
                                        name_scope=name + 'mlp_policy',
                                        name=name + 'mlp_policy',
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
        self.assertTrue(len(policy.parameters('tf_var_list')) == 4)

        ddpg = DDPG(
            env_spec=env_spec,
            config_or_config_dict={
                "REPLAY_BUFFER_SIZE": 10000,
                "GAMMA": 0.999,
                "Q_NET_L1_NORM_SCALE": 0.01,
                "Q_NET_L2_NORM_SCALE": 0.01,
                "CRITIC_LEARNING_RATE": 0.001,
                "ACTOR_LEARNING_RATE": 0.001,
                "DECAY": 0.5,
                "BATCH_SIZE": 50,
                "CRITIC_TRAIN_ITERATION": 1,
                "ACTOR_TRAIN_ITERATION": 1,
                "critic_clip_norm": 0.1,
                "actor_clip_norm": 0.1,
            },
            value_func=mlp_q,
            policy=policy,
            adaptive_learning_rate=True,
            name=name,
            replay_buffer=None
        )
        return ddpg, locals()

    def create_ppo(self, env_id='Swimmer-v1', name='ppo'):
        env = make(env_id)
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
                "value_func_train_batch_size": 10
            },
            value_func=mlp_v,
            stochastic_policy=policy,
            adaptive_learning_rate=True,
            name=name + 'ppo',
        )
        return ppo, locals()

    def create_dict_config(self):
        a = DictConfig(required_key_dict=Foo.required_key_dict,
                       config_dict=dict(var1=1, var2=0.1),
                       cls_name='Foo')
        return a, locals()

    def create_parameters(self):
        parameters = dict(param1='aaaa',
                          param2=12312,
                          param3=np.random.random([4, 2]))
        source_config, _ = self.create_dict_config()
        a = Parameters(parameters=parameters, source_config=source_config,
                       auto_init=False,
                       name='test_params')
        return a, locals()

    def create_continue_dynamics_model(self, env_id='Acrobot-v1', name='mlp_dyna'):
        env = make(env_id)
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        mlp_dyna = ContinuousMLPGlobalDynamicsModel(
            env_spec=env_spec,
            name_scope=name + 'mlp_dyna',
            name=name + 'mlp_dyna',
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
        return mlp_dyna, locals()

    def create_mpc(self, env_id='Acrobot-v1', name='mpc'):
        mlp_dyna, local = self.create_continue_dynamics_model(env_id, name)
        env_spec = local['env_spec']
        env = local['env']
        algo = ModelPredictiveControl(
            dynamics_model=mlp_dyna,
            env_spec=env_spec,
            config_or_config_dict=dict(
                SAMPLED_HORIZON=2,
                SAMPLED_PATH_NUM=5,
                dynamics_model_train_iter=10
            ),
            name=name,
            reward_func=RandomRewardFunc('re_fun'),
            terminal_func=RandomTerminalFunc(name='random_p'),
            policy=UniformRandomPolicy(env_spec=env_spec, name='unp')
        )
        return algo, locals()

    def create_eps(self, env_spec):
        return EpsilonGreedy(action_space=env_spec.action_space,
                             init_random_prob=0.5,
                             decay_type=None), locals()

    def create_agent(self, algo, env, env_spec, eps=None):
        agent = Agent(env=env, env_spec=env_spec,
                      algo=algo,
                      name='agent',
                      exploration_strategy=eps)
        return agent, locals()

    def create_model_free_pipeline(self, env, agent):
        model_free = ModelFreePipeline(agent=agent, env=env,
                                       config_or_config_dict=dict(TEST_SAMPLES_COUNT=100,
                                                                  TRAIN_SAMPLES_COUNT=100,
                                                                  TOTAL_SAMPLES_COUNT=1000))
        return model_free, locals()

    def create_exp(self, pipeline, name):
        experiment = Experiment(
            pipeline=pipeline,
            tuner=None,
            config_or_config_dict={
                "console_logger_to_file_flag": True,
                "console_logger_to_file_name": "console_test.log",
                "log_path": "/home/dls/CAP/mobrl/mobrl/test/tests/tmp_path",
                "log_level": "DEBUG"
            },
            name=name + 'experiment_debug'
        )
        return experiment
