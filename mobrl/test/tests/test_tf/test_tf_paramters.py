import tensorflow as tf
from mobrl.tf.tf_parameters import TensorflowParameters
from mobrl.config.dict_config import DictConfig
from mobrl.core.basic import Basic
from mobrl.test.tests.test_setup import TestTensorflowSetup


class Foo(Basic):
    def __init__(self):
        super().__init__(name='foo')

    required_key_list = dict(var1=1, var2=0.1)


class TestTensorflowParameters(TestTensorflowSetup):
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
