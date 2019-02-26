import unittest
import tensorflow as tf
from mobrl.tf.util import create_new_tf_session
from mobrl.config.global_config import GlobalConfig
import shutil
import os
from mobrl.common.util.logger import Logger, ConsoleLogger
from mobrl.core.global_var import reset_all, get_all
import numpy as np
from mobrl.tf.tf_parameters import TensorflowParameters
from mobrl.config.dict_config import DictConfig
from mobrl.core.basic import Basic
from mobrl.config.global_config import GlobalConfig
from mobrl.tf.util import create_new_tf_session
from mobrl.algo.rl.model_free.dqn import DQN
from mobrl.envs.gym_env import make
from mobrl.envs.env_spec import EnvSpec
from mobrl.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from mobrl.algo.placeholder_input import PlaceholderInput, MultiPlaceholderInput
import glob


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        reset_all()
        for key, val in get_all().items():
            self.assertTrue(len(val) == 0)

    def tearDown(self):
        reset_all()


class TestTensorflowSetup(BaseTestCase):
    default_id = 0

    def setUp(self):
        BaseTestCase.setUp(self)
        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
        tf.reset_default_graph()
        print('set tf device as {}'.format(self.default_id))
        self.sess = create_new_tf_session(cuda_device=self.default_id)

    def tearDown(self):
        if self.sess.run(tf.report_uninitialized_variables()).shape[0] != 0:
            print('some variables are not uninitialized:')
            print(self.sess.run(tf.report_uninitialized_variables()))
            print(self.sess.run(tf.report_uninitialized_variables()).shape)
            raise AssertionError('some variables are not uninitialized')

        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.close()
        BaseTestCase.tearDown(self)

    def assert_var_list_equal(self, var_list1, var_list2):
        for var1, var2 in zip(var_list1, var_list2):
            res1, res2 = self.sess.run([var1, var2])
            self.assertTrue(np.equal(res1, res2).all())

    def assert_var_list_at_least_not_equal(self, var_list1, var_list2):
        res = []
        for var1, var2 in zip(var_list1, var_list2):
            res1, res2 = self.sess.run([var1, var2])
            res.append(np.equal(res1, res2).all())
        self.assertFalse(np.array(res).all())

    def create_dqn(self, env_spec, name):
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
        dqn.init()
        return dqn

    def create_ph(self, name):
        with tf.variable_scope(name):
            a = tf.get_variable(shape=[3, 4], dtype=tf.float32, name='var_1')

        conf = DictConfig(required_key_dict=Foo.required_key_list,
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

        return a


class TestWithLogSet(BaseTestCase):
    def setUp(self):
        BaseTestCase.setUp(self)
        try:
            shutil.rmtree(GlobalConfig.DEFAULT_LOG_PATH)
        except FileNotFoundError:
            pass
        os.mkdir(GlobalConfig.DEFAULT_LOG_PATH)
        self.assertFalse(ConsoleLogger().inited_flag)
        self.assertFalse(Logger().inited_flag)

        Logger().init(config_or_config_dict=GlobalConfig.DEFAULT_LOG_CONFIG_DICT,
                      log_path=GlobalConfig.DEFAULT_LOG_PATH,
                      log_level=GlobalConfig.DEFAULT_LOG_LEVEL)
        ConsoleLogger().init(logger_name='console_logger',
                             to_file_flag=True,
                             level=GlobalConfig.DEFAULT_LOG_LEVEL,
                             to_file_name=os.path.join(GlobalConfig.DEFAULT_LOG_PATH, 'console.log'))

        self.assertTrue(ConsoleLogger().inited_flag)
        self.assertTrue(Logger().inited_flag)

    def tearDown(self):
        # todo a little bug here if to call super()
        BaseTestCase.tearDown(self)
        Logger().reset()
        ConsoleLogger().reset()
        self.assertFalse(ConsoleLogger().inited_flag)
        self.assertFalse(Logger().inited_flag)


class TestWithAll(TestTensorflowSetup, TestWithLogSet):
    def setUp(self):
        TestTensorflowSetup.setUp(self)
        TestWithLogSet.setUp(self)

    def tearDown(self):
        TestWithLogSet.tearDown(self)
        TestTensorflowSetup.tearDown(self)


class SimpleTestSetup(BaseTestCase):
    def setUp(self):
        BaseTestCase.setUp(self)
        try:
            shutil.rmtree(GlobalConfig.DEFAULT_LOG_PATH)
        except FileNotFoundError:
            pass
        os.mkdir(GlobalConfig.DEFAULT_LOG_PATH)
        self.assertFalse(ConsoleLogger().inited_flag)
        self.assertFalse(Logger().inited_flag)

    def tearDown(self):
        BaseTestCase.tearDown(self)
