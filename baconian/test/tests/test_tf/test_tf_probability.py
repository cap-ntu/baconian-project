import tensorflow as tf
import tensorflow_probability as tfp
import unittest
from baconian.core.core import EnvSpec
import numpy as np
from baconian.envs.gym_env import make
from baconian.common.special import *
from baconian.test.tests.set_up.setup import TestTensorflowSetup
import baconian.algo.distribution.mvn as mvn


def describe_sample_tensor_shape(sample_shape, distribution):
    print('Sample shape:', sample_shape)
    print('Returned sample tensor shape:',
          distribution.sample(sample_shape).shape)


def describe_sample_tensor_shapes(distributions, sample_shapes):
    started = False
    for distribution in distributions:
        print(distribution)
        for sample_shape in sample_shapes:
            describe_sample_tensor_shape(sample_shape, distribution)
        print()


def kl_entropy_logprob_from_pat_cody(old_mean, old_var, mean, var, sess, action_dim, action_ph, feed_dict):
    # logvar = tf.reduce_sum(tf.log(var))
    # old_log_var = tf.reduce_sum(tf.log(old_var))
    """
    KL(old|new)
    :param old_mean:
    :param old_var:
    :param mean:
    :param var:
    :param sess:
    :param action_dim:
    :param action_ph:
    :param feed_dict:
    :return:
    """
    logvar = tf.log(var)
    old_log_var = tf.log(old_var)

    log_det_cov_old = tf.reduce_sum(old_log_var)
    log_det_cov_new = tf.reduce_sum(logvar)

    tr_old_new = tf.reduce_sum(tf.exp(old_log_var - logvar))

    kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                              tf.reduce_sum(tf.square(mean - old_mean) /
                                            tf.exp(logvar), axis=1) -
                              action_dim)

    # kl = 0.5 * (log_det_cov_new - log_det_cov_old + tr_old_new +
    #             tf.reduce_sum(tf.square(mean - old_mean) /
    #                           tf.exp(logvar), axis=0) -
    #             action_dim)

    entropy = 0.5 * (action_dim * (np.log(2 * np.pi) + 1) +
                     tf.reduce_sum(logvar))

    logp = -0.5 * tf.reduce_sum(tf.log(tf.exp(logvar) * 2 * np.pi))
    logp += -0.5 * tf.reduce_sum(tf.square(action_ph - mean) /
                                 tf.exp(logvar), axis=1)
    # logp += -0.5 * np.log(2 * np.pi * action_dim)

    logp_old = -0.5 * tf.reduce_sum(tf.log(tf.exp(old_log_var) * 2 * np.pi))
    logp_old += -0.5 * tf.reduce_sum(tf.square(action_ph - old_mean) /
                                     tf.exp(old_log_var), axis=1)
    # logp_old += -0.5 * np.log(2 * np.pi * action_dim)

    return sess.run([kl, entropy, logp, logp_old], feed_dict=feed_dict)


def kl_entropy_logprob_from_mvn(old_mean, old_var, mean, var, sess, action_dim, action_ph, feed_dict):
    kl = mvn.kl(old_mean, old_var, mean, var, action_dim)
    entropy = mvn.entropy(mean, var, action_dim)
    logp = mvn.log_prob(action_ph, mean, var)
    logp_old = mvn.log_prob(action_ph, old_mean, old_var)
    return sess.run([kl, entropy, logp, logp_old], feed_dict=feed_dict)


class TestTFP(TestTensorflowSetup):
    def test_init(self):
        sess = self.sess
        env = make('Pendulum-v0')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        action_dim = env_spec.flat_action_dim
        state_dim = env_spec.flat_obs_dim
        # bs_shape = tf.placeholder(dtype=tf.int8, shape=[])
        bs_shape = 4
        action_ph = tf.placeholder(dtype=tf.float32, shape=[None, action_dim])
        state_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])

        mean_old = tf.layers.dense(inputs=state_ph,
                                   name='layer1',
                                   units=action_dim)

        mean2 = tf.layers.dense(inputs=state_ph,
                                name='layer2',
                                units=action_dim)

        # mean1 = tf.get_variable(name='mean1', shape=[bs_shape, action_dim], dtype=tf.float32)
        var1 = tf.get_variable(name='var1', shape=[action_dim], dtype=tf.float32,
                               initializer=tf.initializers.random_uniform(0.0, 1.0))

        # mean2 = tf.get_variable(name='mean2', shape=[bs_shape, action_dim], dtype=tf.float32)
        var2 = tf.get_variable(name='var2', shape=[action_dim], dtype=tf.float32,
                               initializer=tf.initializers.random_uniform(0.0, 1.0))

        # var1 = tf.get_variable('logvars', (10, action_dim), tf.float32,
        #                        tf.constant_initializer(0.0))
        # var1 = tf.expand_dims(tf.reduce_sum(var1, axis=0), axis=0)
        # var1 = tf.tile(var1, [bs_shape, 1])
        #
        # var2 = tf.get_variable('logvars2', (10, action_dim), tf.float32,
        #                        tf.constant_initializer(0.0))
        # var2 = tf.expand_dims(tf.reduce_sum(var2, axis=0), 0)
        # var2 = tf.tile(var2, [bs_shape, 1])

        dist_old = tfp.distributions.MultivariateNormalDiag(mean_old, tf.sqrt(var1), validate_args=True)
        dis2 = tfp.distributions.MultivariateNormalDiag(mean2, tf.sqrt(var2), validate_args=True)

        dist_norm1 = tfp.distributions.Normal(mean_old, var1)
        dist_norm2 = tfp.distributions.Normal(mean2, var2)

        print(dist_old, dis2)
        # dis1 = tfp.distributions.Independent(dis1, reinterpreted_batch_ndims=1)
        # dis2 = tfp.distributions.Independent(dis2, reinterpreted_batch_ndims=1)

        # op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(tfp.distributions.kl_divergence(dis1, dis2),
        #                                                         var_list=[mean1, var1])

        ac = [env_spec.action_space.sample() for _ in range(bs_shape)]
        ac = make_batch(np.array(ac), original_shape=env_spec.action_shape)

        state = [env_spec.obs_space.sample() for _ in range(bs_shape)]
        state = make_batch(np.array(state), original_shape=env_spec.obs_shape)

        feed_dict = {
            state_ph: state,
            action_ph: ac
        }
        sess.run(tf.global_variables_initializer())

        kl, entropy, logp, log_p_old = kl_entropy_logprob_from_pat_cody(old_mean=mean_old,
                                                                        old_var=var1,
                                                                        mean=mean2,
                                                                        var=var2,
                                                                        feed_dict=feed_dict,
                                                                        sess=sess,
                                                                        action_ph=action_ph,
                                                                        action_dim=action_dim)

        kl_tfp = sess.run(tf.reduce_mean(tfp.distributions.kl_divergence(dist_old, dis2)), feed_dict=feed_dict)
        entropy_tfp = sess.run(tf.reduce_mean(dis2.entropy()), feed_dict=feed_dict)

        log_prob_tfp = sess.run(dis2.log_prob(value=ac), feed_dict=feed_dict)
        log_p_old_tfp = sess.run(dist_old.log_prob(value=ac), feed_dict=feed_dict)

        test_log_prob_tfp = dis2.log_prob(ac) + tf.cast(0.5 * np.log(2. * np.pi * action_dim), dtype=tf.float32)
        test_log_prob_tfp_old = dist_old.log_prob(ac) + tf.cast(0.5 * np.log(2. * np.pi * action_dim), dtype=tf.float32)

        print("ac shape {}".format(ac.shape))
        print("a sample from dis1 shape {}".format(sess.run(dist_old.sample(), feed_dict=feed_dict).shape))
        print("shape of dis under feeddict {}".format(
            sess.run([dist_old.batch_shape_tensor(), dist_old.event_shape_tensor()],
                     feed_dict=feed_dict)))

        # print(sess.run(dis2.log_prob(value=ac)).shape)
        # print(sess.run(dis1.log_prob(value=ac)).shape)
        for i in range(bs_shape):
            feed_dict_i = {
                state_ph: make_batch(state[i], env_spec.obs_shape),
                action_ph: make_batch(ac[i], env_spec.action_shape)
            }
            print("i dis2 log prob: {}".format(sess.run(dis2.log_prob(value=ac[i]), feed_dict=feed_dict_i)))
            print("i dis1 log prob: {}".format(sess.run(dist_old.log_prob(value=ac[i]), feed_dict=feed_dict_i)))

        print(kl, kl_tfp)
        print(entropy, entropy_tfp)

        print(logp, log_prob_tfp)
        print(log_p_old, log_p_old_tfp)
        print('new log p {}'.format(sess.run(test_log_prob_tfp, feed_dict=feed_dict)))
        print('new log p old {}'.format(sess.run(test_log_prob_tfp_old, feed_dict=feed_dict)))

        print('new log p norm {}'.format(sess.run(tf.reduce_sum(dist_norm1.log_prob(ac), axis=1), feed_dict=feed_dict)))
        print('new log p old norm {}'.format(
            sess.run(tf.reduce_sum(dist_norm2.log_prob(ac), axis=1), feed_dict=feed_dict)))

        self.assertTrue(np.isclose(logp, log_prob_tfp).all())
        self.assertTrue(np.isclose(log_p_old, log_p_old_tfp).all())
        self.assertTrue(np.isclose(kl, kl_tfp).all())
        self.assertTrue(np.isclose(entropy, entropy_tfp).all())

        kl, entropy, logp, log_p_old = kl_entropy_logprob_from_mvn(old_mean=mean_old,
                                                                   old_var=var1,
                                                                   mean=mean2,
                                                                   var=var2,
                                                                   feed_dict=feed_dict,
                                                                   sess=sess,
                                                                   action_ph=action_ph,
                                                                   action_dim=action_dim)
        print(kl, entropy, logp, log_p_old)
        self.assertTrue(np.isclose(logp, log_prob_tfp).all())
        self.assertTrue(np.isclose(log_p_old, log_p_old_tfp).all())
        self.assertTrue(np.isclose(entropy, entropy_tfp).all())
        self.assertTrue(np.isclose(kl, kl_tfp).all())
