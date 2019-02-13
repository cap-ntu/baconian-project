import unittest
from mbrl.rl.algo.model_free import DQN
from mbrl.envs.env_spec import EnvSpec
from mbrl.rl.value_func.mlp_q_value import MLPQValueFunction
import tensorflow as tf
from mbrl.tf.util import create_new_tf_session
import numpy as np
from mbrl.tf.tf_parameters import TensorflowParameters
from mbrl.config.dict_config import DictConfig
from mbrl.core.basic import Basic


class Foo(Basic):
    required_key_list = dict(var1=1, var2=0.1)


class TestTensorflowParameters(BaseTestCase):
    def test_tf_param(self):
        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
            # sess.close()
        tf.reset_default_graph()
        sess = create_new_tf_session(cuda_device=0)

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
