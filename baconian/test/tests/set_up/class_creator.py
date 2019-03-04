import numpy as np
import unittest
import tensorflow as tf
from baconian.algo.rl.model_based.models.mlp_dynamics_model import ContinuousMLPGlobalDynamicsModel
from baconian.algo.placeholder_input import PlaceholderInput
from baconian.algo.rl.model_free.dqn import DQN
from baconian.tf.tf_parameters import TensorflowParameters
from baconian.core.core import Basic, EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from baconian.algo.rl.model_free.ddpg import DDPG
from baconian.algo.rl.policy.deterministic_mlp import DeterministicMLPPolicy
from baconian.algo.rl.value_func.mlp_v_value import MLPVValueFunc
from baconian.algo.rl.policy.normal_distribution_mlp import NormalDistributionMLPPolicy
from baconian.algo.rl.model_free.ppo import PPO
from baconian.core.parameters import Parameters, DictConfig
from baconian.algo.rl.model_based.mpc import ModelPredictiveControl
from baconian.algo.rl.model_based.misc.terminal_func.terminal_func import RandomTerminalFunc
from baconian.algo.rl.model_based.misc.reward_func.reward_func import RandomRewardFunc
from baconian.algo.rl.policy.random_policy import UniformRandomPolicy
from baconian.agent.agent import Agent
from baconian.algo.rl.misc.epsilon_greedy import EpsilonGreedy
from baconian.core.pipelines.model_free_pipelines import ModelFreePipeline
from baconian.core.experiment import Experiment
from baconian.core.pipelines.train_test_flow import TrainTestFlow
from baconian.config.global_config import GlobalConfig
from baconian.algo.rl.model_based.sample_with_model import SampleWithDynamics


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
                                     to_ph_parameter_dict=dict(var1=tf.placeholder(shape=(), dtype=tf.int32)))
        return param, locals()

    def create_mlp_q_func(self, env_id='Acrobot-v1', name='mlp_q'):
        env = make(env_id)
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        mlp_q = MLPQValueFunction(env_spec=env_spec,
                                  name_scope=name,
                                  name=name,
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
        return mlp_q, locals()

    def create_dqn(self, env_id='Acrobot-v1', name='dqn'):
        mlp_q, local = self.create_mlp_q_func(env_id, name)
        env_spec = local['env_spec']
        env = local['env']
        dqn = DQN(env_spec=env_spec,
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
                                     to_ph_parameter_dict=dict(var1=tf.placeholder(shape=(), dtype=tf.int32)))
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

    def create_mlp_v(self, env_id='Swimmer-v1', name='mlp_v'):
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
        return mlp_v, locals()

    def create_normal_dist_mlp_policy(self, env_spec, name):
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
        return policy, locals()

    def create_ppo(self, env_id='Swimmer-v1', name='ppo'):
        mlp_v, local = self.create_mlp_v(env_id, name)
        env_spec = local['env_spec']
        env = local['env']
        policy = self.create_normal_dist_mlp_policy(env_spec=env_spec, name=name)[0]
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
            name=name
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
                             init_random_prob=0.5), locals()

    def create_agent(self, algo, env, env_spec, eps=None, name='agent'):
        agent = Agent(env=env, env_spec=env_spec,
                      algo=algo,
                      config_or_config_dict={
                          "TEST_SAMPLES_COUNT": 100,
                          "TRAIN_SAMPLES_COUNT": 100,
                          "TOTAL_SAMPLES_COUNT": 500
                      },
                      name=name,
                      exploration_strategy=eps)
        return agent, locals()

    def create_model_free_pipeline(self, env, agent):
        model_free = ModelFreePipeline(agent=agent, env=env,
                                       config_or_config_dict=dict(TEST_SAMPLES_COUNT=100,
                                                                  TRAIN_SAMPLES_COUNT=100,
                                                                  TOTAL_SAMPLES_COUNT=1000))
        return model_free, locals()

    def create_exp(self, name, env, agent):
        experiment = Experiment(
            tuner=None,
            env=env,
            agent=agent,
            flow=TrainTestFlow(),
            name=name + 'experiment_debug'
        )
        return experiment

    def create_sample_with_model_algo(self, env_spec, model_free_algo, dyanmics_model, name='sample_with_model'):
        algo = SampleWithDynamics(env_spec=env_spec,
                                  name=name,
                                  model_free_algo=model_free_algo,
                                  dynamics_model=dyanmics_model,
                                  config_or_config_dict=dict(
                                      dynamics_model_train_iter=1,
                                      model_free_algo_train_iter=1
                                  ))
        return algo, locals()

    def create_continuous_mlp_global_dynamics_model(self, env_spec, name='continuous_mlp_global_dynamics_model'):
        mlp_dyna = ContinuousMLPGlobalDynamicsModel(
            env_spec=env_spec,
            name_scope=name,
            name=name,
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

    def create_mlp_deterministic_policy(self, name='mlp_policy', env_id='Swimmer-v1'):
        env = make('Swimmer-v1')
        env.reset()
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        policy = DeterministicMLPPolicy(env_spec=env_spec,
                                        name=name,
                                        name_scope=name,
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
                                        output_high=None,
                                        output_low=None,
                                        output_norm=None,
                                        input_norm=None,
                                        reuse=False)
        return policy, locals()
