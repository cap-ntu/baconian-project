from baconian.test.tests.set_up.setup import TestTensorflowSetup
import tensorflow as tf
from baconian.algo.dynamics.dynamics_model import DifferentiableDynamics
import numpy as np


class TestBasicClassInAlgo(TestTensorflowSetup):
    def test_derivable(self):
        val = 10.0
        val2 = 2.0
        bs = 1
        in_node = tf.placeholder(shape=[None, 10], dtype=tf.float32)
        in_node2 = tf.placeholder(shape=[None, 10], dtype=tf.float32)

        out_node = in_node * in_node * val + in_node2 * in_node2 * val2
        a = DifferentiableDynamics(input_node_dict=dict(in_node=in_node, in_node2=in_node2),
                                   output_node_dict=dict(out_node=out_node))
        self.sess.run(tf.global_variables_initializer())
        res = self.sess.run(a.grad_on_input(key_or_node='in_node'), feed_dict={
            in_node: np.random.random([bs, 10]),
            in_node2: np.random.random([bs, 10])
        })
        print('jacobian {}'.format(np.array(res).shape))

        # self.assertTrue(np.equal(res, val).all())

        res = self.sess.run(a.grad_on_input(key_or_node='in_node2'), feed_dict={
            in_node: np.random.random([bs, 10]),
            in_node2: np.random.random([bs, 10])
        })
        print('jacobian {}'.format(np.array(res).shape))

        # self.assertTrue(np.equal(res, val2).all())

        res = self.sess.run(a.grad_on_input(key_or_node=in_node), feed_dict={
            in_node: np.random.random([bs, 10]),
            in_node2: np.random.random([bs, 10])
        })
        # self.assertTrue(np.equal(res, val).all())

        res = self.sess.run(a.grad_on_input(key_or_node=in_node2), feed_dict={
            in_node: np.random.random([bs, 10]),
            in_node2: np.random.random([bs, 10])
        })
        # self.assertTrue(np.equal(res, val2).all())

        res = self.sess.run(a.grad_on_input(key_or_node=in_node, order=2), feed_dict={
            in_node: np.random.random([bs, 10]),
            in_node2: np.random.random([bs, 10])
        })
        print('hessian {}'.format(np.array(res).shape))

        # self.assertTrue(np.equal(res, 0).all())

        res = self.sess.run(a.grad_on_input(key_or_node=in_node2, order=2), feed_dict={
            in_node: np.random.random([bs, 10]),
            in_node2: np.random.random([bs, 10])
        })
        print('hessian {}'.format(np.array(res).shape))

        # self.assertTrue(np.equal(res, 0).all())
