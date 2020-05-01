"""
Module that compute diagonal multivariate normal distribution operation with tensorflow tensor as parameters
"""
import tensorflow as tf
import numpy as np


def kl(mean_p, var_p, mean_q, var_q, dims):
    """
    Compute the KL divergence of diagonal multivariate normal distribution q, and p, which is KL(P||Q)
    :param mean_p:
    :param var_p:
    :param mean_q:
    :param var_q:
    :param dims:
    :return:
    """
    # p is old

    log_var_p = tf.log(var_p)
    log_var_q = tf.log(var_q)
    log_det_cov_p = tf.reduce_sum(log_var_p)
    log_det_cov_q = tf.reduce_sum(log_var_q)
    tr_p_q = tf.reduce_sum(tf.exp(log_var_p - log_var_q))
    kl = 0.5 * tf.reduce_mean(
        log_det_cov_q - log_det_cov_p + tr_p_q + tf.reduce_sum(tf.square(mean_q - mean_p) / tf.exp(log_det_cov_q),
                                                               axis=1) - dims)
    return kl


def entropy(mean_p, var_p, dims):
    return 0.5 * (dims * (np.log(2 * np.pi) + 1) + tf.reduce_sum(tf.log(var_p)))


def log_prob(variable_ph, mean_p, var_p):
    log_prob = -0.5 * (tf.reduce_sum(tf.log(var_p)) + (np.log(2 * np.pi)))
    log_prob += -0.5 * tf.reduce_sum(tf.square(variable_ph - mean_p) / var_p, axis=1)
    return log_prob


def prob(variable_ph, mean_p, var_p):
    return tf.log(log_prob(variable_ph, mean_p, var_p))


def sample(mean_p, var_p, dims):
    return mean_p + tf.math.sqrt(var_p) * tf.random_normal(shape=(dims,))
