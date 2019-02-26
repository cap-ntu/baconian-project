import numpy as np
import tensorflow as tf
from mobrl.tf.tf_parameters import TensorflowParameters
from mobrl.config.dict_config import DictConfig
from mobrl.core.basic import Basic
from mobrl.test.tests.test_setup import TestWithAll
from mobrl.config.global_config import GlobalConfig
from mobrl.tf.util import create_new_tf_session
from mobrl.algo.rl.model_free.dqn import DQN
from mobrl.envs.gym_env import make
from mobrl.envs.env_spec import EnvSpec
from mobrl.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from mobrl.test.tests.test_setup import TestTensorflowSetup


class Foo(Basic):
    def __init__(self):
        super().__init__(name='foo')

    required_key_list = dict(var1=1, var2=0.1)


class TestTensorflowParameters(TestWithAll):
    def test_tf_param(self):
        with tf.variable_scope('test_tf_param'):
            a = tf.get_variable(shape=[3, 4], dtype=tf.float32, name='var_1')
            b = tf.get_variable(shape=[3, 4], dtype=tf.bool, name='var_2')

        conf = DictConfig(required_key_dict=Foo.required_key_list,
                          config_dict=dict(var1=1, var2=0.01))
        param = TensorflowParameters(tf_var_list=[a, b],
                                     rest_parameters=dict(var3='sss'),
                                     name='temp',
                                     source_config=conf,
                                     require_snapshot=True,
                                     to_ph_parameter_dict=dict(var1=tf.placeholder(shape=(), dtype=tf.int32)),
                                     auto_init=False)
        param.init()
        param.save_snapshot()
        param.load_snapshot()

    def test_save_load(self):

        with tf.variable_scope('test_tf_param'):
            a = tf.get_variable(shape=[3, 4], dtype=tf.float32, name='var_1')
            b = tf.get_variable(shape=[3, 4], dtype=tf.bool, name='var_2')

        conf = DictConfig(required_key_dict=Foo.required_key_list,
                          config_dict=dict(var1=1, var2=0.01))
        param = TensorflowParameters(tf_var_list=[a, b],
                                     rest_parameters=dict(var3='sss'),
                                     name='temp',
                                     source_config=conf,
                                     require_snapshot=True,
                                     to_ph_parameter_dict=dict(var1=tf.placeholder(shape=(), dtype=tf.int32)),
                                     auto_init=False)
        param.init()
        var_val = [self.sess.run(var) for var in param('tf_var_list')]

        with tf.variable_scope('other_test_tf_param'):
            a = tf.get_variable(shape=[3, 4], dtype=tf.float32, name='var_1')
            b = tf.get_variable(shape=[3, 4], dtype=tf.bool, name='var_2')

        conf = DictConfig(required_key_dict=Foo.required_key_list,
                          config_dict=dict(var1=1, var2=0.01))
        param_other = TensorflowParameters(tf_var_list=[a, b],
                                           rest_parameters=dict(var3='sss'),
                                           name='tempother',
                                           source_config=conf,
                                           require_snapshot=True,
                                           to_ph_parameter_dict=dict(var1=tf.placeholder(shape=(), dtype=tf.int32)),
                                           auto_init=False)
        param_other.init()

        for i in range(10):
            param.save(sess=self.sess, save_path=GlobalConfig.DEFAULT_LOG_PATH + '/model', global_step=i)

        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
        tf.reset_default_graph()
        print('set tf device as {}'.format(self.default_id))
        self.sess = create_new_tf_session(cuda_device=self.default_id)

        with tf.variable_scope('test_tf_param'):
            a = tf.get_variable(shape=[3, 4], dtype=tf.float32, name='var_1')
            b = tf.get_variable(shape=[3, 4], dtype=tf.bool, name='var_2')

        conf = DictConfig(required_key_dict=Foo.required_key_list,
                          config_dict=dict(var1=1, var2=0.01))
        param2 = TensorflowParameters(tf_var_list=[a, b],
                                      rest_parameters=dict(var3='sss'),
                                      name='temp2',
                                      source_config=conf,
                                      require_snapshot=True,
                                      to_ph_parameter_dict=dict(var1=tf.placeholder(shape=(), dtype=tf.int32)),
                                      auto_init=False)
        param2.init()
        param2.load(path_to_model=GlobalConfig.DEFAULT_LOG_PATH + '/model', model_name='temp',
                    global_step=None)
        for var1, var2 in zip(var_val, param2('tf_var_list')):
            self.assertTrue(np.equal(var1, self.sess.run(var2)).all())

    def test_save_load_with_dqn(self):
        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        mlp_q = MLPQValueFunction(env_spec=env_spec,
                                  name_scope='mlp_q',
                                  name='mlp_q',
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
                  value_func=mlp_q)
        dqn.init()
