import unittest
import tensorflow as tf
from baconian.algo.dynamics.mlp_dynamics_model import ContinuousMLPGlobalDynamicsModel
from baconian.algo.misc.placeholder_input import PlaceholderInput
from baconian.algo.dqn import DQN
from baconian.tf.tf_parameters import ParametersWithTensorflowVariable
from baconian.core.core import Basic, EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.algo.ddpg import DDPG
from baconian.algo.policy import DeterministicMLPPolicy
from baconian.algo.policy import ConstantActionPolicy
from baconian.algo.value_func import MLPVValueFunc
from baconian.algo.policy.normal_distribution_mlp import NormalDistributionMLPPolicy
from baconian.algo.ppo import PPO
from baconian.core.parameters import Parameters, DictConfig
from baconian.algo.mpc import ModelPredictiveControl
from baconian.algo.dynamics.terminal_func.terminal_func import RandomTerminalFunc
from baconian.algo.dynamics.reward_func.reward_func import RandomRewardFunc, CostFunc
from baconian.algo.policy import UniformRandomPolicy
from baconian.core.agent import Agent
from baconian.algo.misc import EpsilonGreedy
from baconian.core.experiment import Experiment
from baconian.core.flow.train_test_flow import TrainTestFlow
from baconian.algo.dyna import Dyna
from baconian.common.schedules import *
from baconian.core.status import *
from baconian.algo.policy.ilqr_policy import iLQRPolicy
from baconian.algo.dynamics.random_dynamics_model import UniformRandomDynamicsModel
from baconian.common.noise import *
from baconian.core.flow.dyna_flow import DynaFlow
from baconian.common.data_pre_processing import *
from baconian.common.sampler.sample_data import TransitionData, TrajectoryData


class Foo(Basic):
    def __init__(self):
        super().__init__(name='foo')

    required_key_dict = dict(var1=1, var2=0.1)


class ClassCreatorSetup(unittest.TestCase):

    def create_env(self, env_id):
        return make(env_id)

    def create_env_spec(self, env):
        return EnvSpec(action_space=env.action_space,
                       obs_space=env.observation_space)

    def create_tf_parameters(self, name='test_tf_param'):
        with tf.variable_scope(name):
            a = tf.get_variable(shape=[3, 4], dtype=tf.float32, name='var_1')
            b = tf.get_variable(shape=[3, 4], dtype=tf.bool, name='var_2')

        conf = DictConfig(required_key_dict=Foo.required_key_dict,
                          config_dict=dict(var1=1, var2=0.01))
        param = ParametersWithTensorflowVariable(tf_var_list=[a, b],
                                                 rest_parameters=dict(var3='sss'),
                                                 name=name,
                                                 source_config=conf,
                                                 require_snapshot=True,
                                                 to_ph_parameter_dict=dict(
                                                     var1=tf.placeholder(shape=(), dtype=tf.int32)))
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
                                          "W_NORMAL_STDDEV": 0.03,
                                          "L1_NORM": 0.2,
                                          "L2_NORM": 0.1
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
        mlp_q, local = self.create_mlp_q_func(env_id, name='{}_mlp_q'.format(name))
        env_spec = local['env_spec']
        env = local['env']
        dqn = DQN(env_spec=env_spec,
                  config_or_config_dict=dict(REPLAY_BUFFER_SIZE=1000,
                                             GAMMA=0.99,
                                             BATCH_SIZE=10,
                                             LEARNING_RATE=0.001,
                                             TRAIN_ITERATION=1,
                                             DECAY=0.5),
                  name=name,
                  value_func=mlp_q)
        return dqn, locals()

    def create_ph(self, name):
        with tf.variable_scope(name):
            a = tf.get_variable(shape=[3, 4], dtype=tf.float32, name='var_1')

        conf = DictConfig(required_key_dict=Foo.required_key_dict,
                          config_dict=dict(var1=1, var2=0.01))
        param = ParametersWithTensorflowVariable(tf_var_list=[a],
                                                 rest_parameters=dict(var3='sss'),
                                                 name=name,
                                                 source_config=conf,
                                                 require_snapshot=True,
                                                 to_ph_parameter_dict=dict(
                                                     var1=tf.placeholder(shape=(), dtype=tf.int32)))
        param.init()
        a = PlaceholderInput(parameters=param)

        return a, locals()

    def create_ddpg(self, env_id='Pendulum-v0', name='ddpg'):
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
            name=name,
            replay_buffer=None
        )
        return ddpg, locals()

    def create_mlp_v(self, env_id='Pendulum-v0', name='mlp_v'):
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
        return mlp_v, locals()

    def create_normal_dist_mlp_policy(self, env_spec, name='norm_dist_p_'):
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
        return policy, locals()

    def create_ppo(self, env_id='Pendulum-v0', name='ppo'):
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
                "value_func_train_batch_size": 10,
                "lr_multiplier": 1.0
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

        mlp_dyna, _ = self.create_continuous_mlp_global_dynamics_model(env_spec=env_spec, name=name)
        return mlp_dyna, locals()

    def create_mpc(self, env_id='Acrobot-v1', name='mpc', policy=None, mlp_dyna=None, env_spec=None, env=None):
        if mlp_dyna is None:
            mlp_dyna, local = self.create_continue_dynamics_model(env_id, name + 'mlp_dyna')
            env_spec = local['env_spec']
            env = local['env']

        policy = policy if policy else UniformRandomPolicy(env_spec=env_spec, name='unp')

        algo = ModelPredictiveControl(
            dynamics_model=mlp_dyna,
            env_spec=env_spec,
            config_or_config_dict=dict(
                SAMPLED_HORIZON=2,
                SAMPLED_PATH_NUM=5,
                dynamics_model_train_iter=10
            ),
            name=name,
            policy=policy
        )
        algo.set_terminal_reward_function_for_dynamics_env(terminal_func=RandomTerminalFunc(name='random_p'),
                                                           reward_func=RandomRewardFunc('re_fun'))
        return algo, locals()

    def create_eps(self, env_spec):
        return EpsilonGreedy(action_space=env_spec.action_space,
                             init_random_prob=0.5), locals()

    def create_agent(self, algo, env, env_spec, eps=None, name='agent'):
        agent = Agent(env=env,
                      env_spec=env_spec,
                      algo=algo,
                      noise_adder=AgentActionNoiseWrapper(noise=OUNoise(),
                                                          action_weight_scheduler=LinearScheduler(
                                                              t_fn=lambda: get_global_status_collect()(
                                                                  'TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                                                              schedule_timesteps=100,
                                                              final_p=1.0,
                                                              initial_p=0.0),
                                                          noise_weight_scheduler=LinearScheduler(
                                                              t_fn=lambda: get_global_status_collect()(
                                                                  'TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                                                              schedule_timesteps=100,
                                                              final_p=0.0,
                                                              initial_p=1.0)),
                      name=name,
                      algo_saving_scheduler=PeriodicalEventSchedule(
                          t_fn=lambda: get_global_status_collect()('TOTAL_ENV_STEP_TRAIN_SAMPLE_COUNT'),
                          trigger_every_step=20,
                          after_t=10),
                      exploration_strategy=eps)
        return agent, locals()

    def create_train_test_flow(self, agent, traj_flag=False):

        flow = TrainTestFlow(
            train_sample_count_func=lambda: get_global_status_collect()('TOTAL_ENV_STEP_TRAIN_SAMPLE_COUNT'),
            config_or_config_dict={
                "TEST_EVERY_SAMPLE_COUNT": 10,
                "TRAIN_EVERY_SAMPLE_COUNT": 10,
                "START_TRAIN_AFTER_SAMPLE_COUNT": 5,
                "START_TEST_AFTER_SAMPLE_COUNT": 5,
            },
            func_dict={
                'test': {'func': agent.test,
                         'args': list(),
                         'kwargs': dict(sample_count=1),
                         },
                'train': {'func': agent.train,
                          'args': list(),
                          'kwargs': dict(),
                          },
                'sample': {'func': agent.sample,
                           'args': list(),
                           'kwargs': dict(sample_count=100 if not traj_flag else 1,
                                          env=agent.env,
                                          sample_type='trajectory' if traj_flag else 'transition',
                                          in_which_status='TRAIN',
                                          store_flag=True),
                           },
            }
        )
        return flow

    def create_dyna_flow(self, agent, env):
        flow = DynaFlow(
            train_sample_count_func=lambda: get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
            config_or_config_dict={
                "TEST_ALGO_EVERY_REAL_SAMPLE_COUNT": 10,
                "TEST_DYNAMICS_EVERY_REAL_SAMPLE_COUNT": 10,
                "TRAIN_ALGO_EVERY_REAL_SAMPLE_COUNT_FROM_REAL_ENV": 10,
                "TRAIN_ALGO_EVERY_REAL_SAMPLE_COUNT_FROM_DYNAMICS_ENV": 10,
                "TRAIN_DYNAMICS_EVERY_REAL_SAMPLE_COUNT": 10,
                "START_TRAIN_ALGO_AFTER_SAMPLE_COUNT": 1,
                "START_TRAIN_DYNAMICS_AFTER_SAMPLE_COUNT": 1,
                "START_TEST_ALGO_AFTER_SAMPLE_COUNT": 1,
                "START_TEST_DYNAMICS_AFTER_SAMPLE_COUNT": 1,
                "WARM_UP_DYNAMICS_SAMPLES": 1
            },
            func_dict={
                'train_algo': {'func': agent.train,
                               'args': list(),
                               'kwargs': dict(state='state_agent_training')},
                'train_algo_from_synthesized_data': {'func': agent.train,
                                                     'args': list(),
                                                     'kwargs': dict(state='state_agent_training')},
                'train_dynamics': {'func': agent.train,
                                   'args': list(),
                                   'kwargs': dict(state='state_dynamics_training')},
                'test_algo': {'func': agent.test,
                              'args': list(),
                              'kwargs': dict(sample_count=10)},
                'test_dynamics': {'func': agent.algo.test_dynamics,
                                  'args': list(),
                                  'kwargs': dict(sample_count=10, env=env)},
                'sample_from_real_env': {'func': agent.sample,
                                         'args': list(),
                                         'kwargs': dict(sample_count=10,
                                                        env=agent.env,
                                                        in_which_status='TRAIN',
                                                        store_flag=True)},
                'sample_from_dynamics_env': {'func': agent.sample,
                                             'args': list(),
                                             'kwargs': dict(sample_count=10,
                                                            env=agent.algo.dynamics_env,
                                                            in_which_status='TRAIN',
                                                            store_flag=True)}
            }
        )
        return flow, locals()

    def create_exp(self, name, env, agent, flow=None, traj_flag=False):
        experiment = Experiment(
            tuner=None,
            env=env,
            agent=agent,
            flow=self.create_train_test_flow(agent, traj_flag=traj_flag) if not flow else flow,
            name=name + 'experiment_debug'
        )
        return experiment

    def create_dyna(self, env_spec=None, model_free_algo=None, dyanmics_model=None,
                    name='dyna'):
        if not env_spec:
            model_free_algo, local = self.create_ddpg()
            dyanmics_model, _ = self.create_continuous_mlp_global_dynamics_model(env_spec=local['env_spec'])
            env_spec = local['env_spec']
            env = local['env']
        algo = Dyna(env_spec=env_spec,
                    name=name,
                    model_free_algo=model_free_algo,
                    dynamics_model=dyanmics_model,
                    config_or_config_dict=dict(
                        dynamics_model_train_iter=1,
                        model_free_algo_train_iter=1
                    ))
        algo.set_terminal_reward_function_for_dynamics_env(terminal_func=RandomTerminalFunc(),
                                                           reward_func=RandomRewardFunc())
        return algo, locals()

    def create_continuous_mlp_global_dynamics_model(self, env_spec, name='continuous_mlp_global_dynamics_model'):
        mlp_dyna = ContinuousMLPGlobalDynamicsModel(
            env_spec=env_spec,
            name_scope=name,
            name=name,
            state_input_scaler=RunningStandardScaler(dims=env_spec.flat_obs_dim),
            action_input_scaler=RunningStandardScaler(dims=env_spec.flat_action_dim),
            output_delta_state_scaler=RunningStandardScaler(dims=env_spec.flat_obs_dim),
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

    def create_mlp_deterministic_policy(self, env_spec, name='mlp_policy'):
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

    def create_uniform_policy(self, env_spec, name='uni_policy'):

        return UniformRandomPolicy(env_spec=env_spec, name=name), locals()

    def create_constant_action_policy(self, env_spec, name='constant_policy'):
        return ConstantActionPolicy(env_spec=env_spec,
                                    name=name,
                                    config_or_config_dict=dict(
                                        ACTION_VALUE=np.array(env_spec.action_space.sample()))), locals()

    def create_ilqr_policy(self, env_id='Pendulum-v0'):

        class DebuggingCostFunc(CostFunc):
            def __call__(self, state=None, action=None, new_state=None, **kwargs) -> float:
                return float(np.sum(action * action))

        env = make(env_id)
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        dyna = UniformRandomDynamicsModel(env_spec=env_spec)
        dyna.init()
        policy = iLQRPolicy(env_spec=env_spec,
                            T=50,
                            delta=0.0005,
                            iteration=5,
                            dynamics=dyna,
                            dynamics_model_train_iter=1,
                            cost_fn=DebuggingCostFunc())
        return policy, locals()

    def sample_transition(self, env, count=100):
        data = TransitionData(env.env_spec)
        st = env.get_state()
        for i in range(count):
            ac = env.env_spec.action_space.sample()
            new_st, re, done, info = env.step(action=ac)
            data.append(state=st,
                        action=ac,
                        new_state=new_st,
                        done=done,
                        reward=re)
        return data

    def register_global_status_when_test(self, agent, env):
        """
        this func should be called only for unit test
        :param agent:
        :param env:
        :return:
        """
        get_global_status_collect().register_info_key_status(obj=agent,
                                                             info_key='predict_counter',
                                                             under_status='TRAIN',
                                                             return_name='TOTAL_AGENT_TRAIN_SAMPLE_COUNT')
        get_global_status_collect().register_info_key_status(obj=agent,
                                                             info_key='predict_counter',
                                                             under_status='TEST',
                                                             return_name='TOTAL_AGENT_TEST_SAMPLE_COUNT')
        get_global_status_collect().register_info_key_status(obj=agent,
                                                             info_key='update_counter',
                                                             under_status='TRAIN',
                                                             return_name='TOTAL_AGENT_UPDATE_COUNT')
        get_global_status_collect().register_info_key_status(obj=env,
                                                             info_key='step',
                                                             under_status='TEST',
                                                             return_name='TOTAL_ENV_STEP_TEST_SAMPLE_COUNT')
        get_global_status_collect().register_info_key_status(obj=env,
                                                             info_key='step',
                                                             under_status='TRAIN',
                                                             return_name='TOTAL_ENV_STEP_TRAIN_SAMPLE_COUNT')
