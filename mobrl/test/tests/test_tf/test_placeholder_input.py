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
import glob


class Foo(Basic):
    def __init__(self):
        super().__init__(name='foo')

    required_key_list = dict(var1=1, var2=0.1)


class TestPlaceholderInput(TestWithAll):
    def test_tf_param(self):
        a = self.create_ph('test')
        for i in range(5):
            a.save(save_path=GlobalConfig.DEFAULT_LOG_PATH + '/test_placehoder_input',
                   global_step=i,
                   name='a')
        file = glob.glob(GlobalConfig.DEFAULT_LOG_PATH + '/test_placehoder_input/a*.meta')
        self.assertTrue(len(file) == 5)
        b = self.create_ph('b')
        b.copy_from(obj=a)
        self.assert_var_list_equal(a.parameters('tf_var_list'),
                                   b.parameters('tf_var_list'))

        a.parameters.init()
        self.assert_var_list_at_least_not_equal(a.parameters('tf_var_list'),
                                                b.parameters('tf_var_list'))

        a.load(path_to_model=GlobalConfig.DEFAULT_LOG_PATH + '/test_placehoder_input',
               global_step=4,
               model_name='a')

        self.assert_var_list_equal(a.parameters('tf_var_list'),
                                   b.parameters('tf_var_list'))

    def test_save_load_with_dqn(self):
        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        dqn = self.create_dqn(env_spec, 'test_')

        for i in range(5):
            dqn.save(save_path=GlobalConfig.DEFAULT_LOG_PATH + '/test_placehoder_input', global_step=i, name='dqn')
        file = glob.glob(GlobalConfig.DEFAULT_LOG_PATH + '/test_placehoder_input/dqn*.meta')
        self.assertTrue(len(file) == 5)
        dqn2 = self.create_dqn(env_spec, 'test2')
        dqn2.copy_from(dqn)

        self.assert_var_list_equal(dqn.parameters('tf_var_list'), dqn2.parameters('tf_var_list'))
        self.assert_var_list_equal(dqn.q_value_func.parameters('tf_var_list'),
                                   dqn2.q_value_func.parameters('tf_var_list'))
        self.assert_var_list_equal(dqn.target_q_value_func.parameters('tf_var_list'),
                                   dqn2.target_q_value_func.parameters('tf_var_list'))

        dqn.init()
        # self.assert_var_list_at_least_not_equal(dqn.parameters('tf_var_list'), dqn2.parameters('tf_var_list'))
        self.assert_var_list_at_least_not_equal(dqn.q_value_func.parameters('tf_var_list'),
                                                dqn2.q_value_func.parameters('tf_var_list'))
        self.assert_var_list_at_least_not_equal(dqn.target_q_value_func.parameters('tf_var_list'),
                                                dqn2.target_q_value_func.parameters('tf_var_list'))
        dqn.load(path_to_model=GlobalConfig.DEFAULT_LOG_PATH + '/test_placehoder_input', global_step=4,
                 model_name='dqn')

        self.assert_var_list_equal(dqn.parameters('tf_var_list'), dqn2.parameters('tf_var_list'))
        self.assert_var_list_equal(dqn.q_value_func.parameters('tf_var_list'),
                                   dqn2.q_value_func.parameters('tf_var_list'))
        self.assert_var_list_equal(dqn.target_q_value_func.parameters('tf_var_list'),
                                   dqn2.target_q_value_func.parameters('tf_var_list'))
