import unittest
import numpy as np
from baconian.tf.util import MLPCreator
from baconian.test.tests.set_up.setup import TestTensorflowSetup
import tensorflow as tf


class TestTensorflowUtil(TestTensorflowSetup):

    def test_init_with_tf_layers(self):
        input_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='ph1')
        input_ph2 = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='ph1')
        net1_name = 'net'
        net1_scope = 'scope'

        net2_name = 'net'
        net2_scope = 'scope'

        net1 = MLPCreator.create_network_with_tf_layers(input=input_ph,
                                                        network_config=[
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
                                                        ],
                                                        net_name=net1_name,
                                                        reuse=False,
                                                        tf_var_scope=net1_scope)

        net2 = MLPCreator.create_network_with_tf_layers(input=input_ph,
                                                        network_config=[
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
                                                        ],
                                                        net_name=net2_name,
                                                        reuse=True,
                                                        tf_var_scope=net2_scope)
        self.assertGreater(len(net1[2]), 0)
        self.assertGreater(len(net2[2]), 0)
        self.sess.run(tf.global_variables_initializer())
        for var1, var2 in zip(net1[2], net2[2]):
            print("net1: {} {}  | true name: {}".format(net1_scope, net1_name, var1.name))
            print("net2: {} {} | true name: {}".format(net2_scope, net2_name, var2.name))
            self.assertEqual(id(var1), id(var2))
            self.assertTrue(net1_name in var1.name)
            self.assertTrue(net1_scope in var1.name)
            self.assertTrue(net2_name in var2.name)
            self.assertTrue(net2_scope in var2.name)

        var = net1[2][0]
        op = tf.assign(var,
                       tf.constant(value=np.random.random(list(self.sess.run(tf.shape(var)))),
                                   dtype=tf.float32))
        self.sess.run(op)
        var1 = self.sess.run(var)
        var2 = self.sess.run(net2[2][0])
        self.assertTrue(np.equal(var1, var2).all())

    def test_init_with_tf_layers_2(self):
        input_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='ph1')
        net1_name = 'net'
        net1_scope = 'scope'

        net2_name = 'net'
        net2_scope = 'scope1'

        net1 = MLPCreator.create_network_with_tf_layers(input=input_ph,
                                                        network_config=[
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
                                                        ],
                                                        net_name=net1_name,
                                                        reuse=False,
                                                        tf_var_scope=net1_scope)

        net2 = MLPCreator.create_network_with_tf_layers(input=input_ph,
                                                        network_config=[
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
                                                        ],
                                                        net_name=net2_name,
                                                        reuse=False,
                                                        tf_var_scope=net2_scope)
        self.assertGreater(len(net1[2]), 0)
        self.assertGreater(len(net2[2]), 0)
        self.sess.run(tf.global_variables_initializer())
        for var1, var2 in zip(net1[2], net2[2]):
            print("net1: {} {}  | true name: {}".format(net1_scope, net1_name, var1.name))
            print("net2: {} {} | true name: {}".format(net2_scope, net2_name, var2.name))
            self.assertFalse(id(var1) == id(var2))
            self.assertTrue(net1_name in var1.name)
            self.assertTrue(net1_scope in var1.name)
            self.assertTrue(net2_name in var2.name)
            self.assertTrue(net2_scope in var2.name)
        var = net1[2][0]
        op = tf.assign(var,
                       tf.constant(value=np.random.random(list(self.sess.run(tf.shape(var)))),
                                   dtype=tf.float32))
        self.sess.run(op)
        var1 = self.sess.run(var)
        var2 = self.sess.run(net2[2][0])
        self.assertFalse(np.equal(var1, var2).all())

    def test_init_with_tf_layers_3(self):
        input_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='ph1')
        net1_name = 'net'
        net1_scope = 'scope'

        net2_name = 'net'
        net2_scope = 'scope'

        net1 = MLPCreator.create_network_with_tf_layers(input=input_ph,
                                                        network_config=[
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
                                                        ],
                                                        net_name=net1_name,
                                                        reuse=False,
                                                        tf_var_scope=net1_scope)
        input_ph2 = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='ph1')
        input_ph2 = tf.tanh(input_ph2)
        net2 = MLPCreator.create_network_with_tf_layers(input=input_ph2,
                                                        network_config=[
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
                                                        ],
                                                        net_name=net2_name,
                                                        reuse=True,
                                                        tf_var_scope=net2_scope)
        self.assertGreater(len(net1[2]), 0)
        self.assertGreater(len(net2[2]), 0)
        self.assertEqual(len(net1), len(net2))
        self.sess.run(tf.global_variables_initializer())
        for var1, var2 in zip(net1[2], net2[2]):
            print("net1: {} {}  | true name: {}".format(net1_scope, net1_name, var1.name))
            print("net2: {} {} | true name: {}".format(net2_scope, net2_name, var2.name))
            self.assertTrue(id(var1) == id(var2))
            self.assertTrue(net1_name in var1.name)
            self.assertTrue(net1_scope in var1.name)
            self.assertTrue(net2_name in var2.name)
            self.assertTrue(net2_scope in var2.name)
        var = net1[2][0]
        op = tf.assign(var,
                       tf.constant(value=np.random.random(list(self.sess.run(tf.shape(var)))),
                                   dtype=tf.float32))
        self.sess.run(op)
        var1 = self.sess.run(var)
        var2 = self.sess.run(net2[2][0])
        self.assertTrue(np.equal(var1, var2).all())


if __name__ == '__main__':
    unittest.main()
