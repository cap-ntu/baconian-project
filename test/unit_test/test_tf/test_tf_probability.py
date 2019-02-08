import tensorflow as tf
import tensorflow_probability as tfp
import unittest
import unittest
from src.rl.algo.model_free import DQN
from src.envs.env_spec import EnvSpec
from src.rl.value_func.mlp_q_value import MLPQValueFunction
from src.tf.util import create_new_tf_session
import numpy as np
from src.tf.tf_parameters import TensorflowParameters
from src.config.dict_config import DictConfig
from src.core.basic import Basic
from src.tf.util import MLPCreator


def kl_entropy_from_pat_cody(old_mean, old_var, mean, var, sess, action_dim):
    logvar = tf.log(var)
    old_log_var = tf.log(old_var)

    log_det_cov_old = tf.reduce_sum(old_log_var)
    log_det_cov_new = tf.reduce_sum(logvar)

    tr_old_new = tf.reduce_sum(tf.exp(old_log_var - logvar))

    kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                              tf.reduce_sum(tf.square(mean - old_mean) /
                                            tf.exp(logvar), axis=0) -
                              action_dim)
    entropy = 0.5 * (action_dim * (np.log(2 * np.pi) + 1) +
                     tf.reduce_sum(logvar))

    return sess.run([kl, entropy])


class TestTFP(unittest.TestCase):
    def test_init(self):
        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
        tf.reset_default_graph()
        sess = create_new_tf_session(cuda_device=0)
        action_dim = 5
        mean1 = tf.get_variable(name='mean1', shape=[action_dim], dtype=tf.float32)
        var1 = tf.get_variable(name='var1', shape=[action_dim], dtype=tf.float32,
                               initializer=tf.initializers.random_uniform(0.0, 1.0))

        mean2 = tf.get_variable(name='mean2', shape=[action_dim], dtype=tf.float32)
        var2 = tf.get_variable(name='var2', shape=[action_dim], dtype=tf.float32,
                               initializer=tf.initializers.random_uniform(0.0, 1.0))

        dis1 = tfp.distributions.MultivariateNormalDiag(loc=mean1, scale_diag=var1)
        dis2 = tfp.distributions.MultivariateNormalDiag(loc=mean2, scale_diag=var2)
        op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(tfp.distributions.kl_divergence(dis1, dis2),
                                                                var_list=[mean1, var1])

        sess.run(tf.global_variables_initializer())

        # print(sess.run(dis1.sample()))
        # print(sess.run(dis2.sample()))
        # print(sess.run(tfp.distributions.kl_divergence(dis1, dis2)))
        # print(sess.run(tfp.distributions.kl_divergence(dis2, dis1)))
        # print(sess.run(dis2.cross_entropy(dis2)))
        # print(sess.run(dis1.cross_entropy(dis1)))
        # print(sess.run(dis1.cross_entropy(dis2)))
        # print(sess.run(dis2.cross_entropy(dis1)))

        print(sess.run([dis1.mean(), dis2.mean()]))
        print(sess.run([dis1.variance(), dis2.variance()]))

        kl, entropy = kl_entropy_from_pat_cody(old_mean=dis1.mean(),
                                               old_var=dis1.variance(),
                                               mean=dis2.mean(),
                                               var=dis2.variance(),
                                               sess=sess,
                                               action_dim=action_dim)

        kl_tfp = sess.run(tf.reduce_mean(tfp.distributions.kl_divergence(dis1, dis2)))
        entropy_tfp = sess.run(tf.reduce_mean(dis2.entropy()))

        print(kl, entropy)
        print(kl_tfp, entropy_tfp)

        self.assertTrue(np.isclose(kl, kl_tfp).all())
        self.assertTrue(np.isclose(entropy, entropy_tfp).all())
