import os

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer as contrib_W_init
from typeguard import typechecked
import collections
import multiprocessing
import tensorflow.contrib as tf_contrib
from baconian.common.error import *

__all__ = ['get_tf_collection_var_list', 'MLPCreator']


def get_tf_collection_var_list(scope, key=tf.GraphKeys.GLOBAL_VARIABLES):
    var_list = tf.get_collection(key, scope=scope)
    return sorted(list(set(var_list)), key=lambda x: x.name)


# def create_new_tf_session(cuda_device: int):
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
#     tf_config = tf.ConfigProto()
#     tf_config.gpu_options.allow_growth = True
#     sess = tf.Session(config=tf_config)
#     sess.__enter__()
#     assert tf.get_default_session()
#     return sess

def clip_grad(optimizer, loss, clip_norm: float, var_list):
    grad_var_pair = optimizer.compute_gradients(loss=loss, var_list=var_list)
    if clip_norm <= 0.0:
        raise InappropriateParameterSetting('clip_norm should be larger than 0.0')
    grad_var_pair = [(tf.clip_by_norm(grad, clip_norm=clip_norm), var) for
                     grad, var in grad_var_pair]
    grad = [g[0] for g in grad_var_pair]
    return grad_var_pair, grad


def create_new_tf_session(**kwargs):
    """Get default session or create one with a given config"""
    sess = tf.get_default_session()
    if sess is None:
        sess = make_session(**kwargs)
    sess.__enter__()
    assert tf.get_default_session()
    return sess


def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)


# class TensorInput(object):
#     def __init__(self, **kwargs):
#         for key, val in kwargs:
#             setattr(self, key, val)


class MLPCreator(object):
    act_dict = {
        'LINEAR': tf.identity,
        'RELU': tf.nn.relu,
        'LEAKY_RELU': tf.nn.leaky_relu,
        'SIGMOID': tf.nn.sigmoid,
        'SOFTMAX': tf.nn.softmax,
        'IDENTITY': tf.identity,
        'TANH': tf.nn.tanh,
        'ELU': tf.nn.elu
    }

    @staticmethod
    def create_network_with_tf_layers(input: tf.Tensor, network_config: list, tf_var_scope: str, net_name='',
                                      input_norm=None,
                                      output_norm=None,
                                      reuse=False,
                                      output_low=None, output_high=None):
        """
        Create a MLP network with a input tensor
        warning: this will create a input net which will cut the gradients from the input tensor and its
        previous op
        :param input:
        :param network_config:
        :param net_name:
        :param tf_var_scope:
        :param input_norm:
        :param output_norm:
        :param output_low:
        :param output_high:
        :return:
        """
        pre_var_scope_name = tf.get_variable_scope().name
        tf_var_scope_context = tf.variable_scope(tf_var_scope)
        tf_var_scope_context.__enter__()
        if pre_var_scope_name != '':
            assert tf.get_variable_scope().name == "{}/{}".format(pre_var_scope_name, tf_var_scope)
        else:
            assert tf.get_variable_scope().name == "{}".format(tf_var_scope)

        if reuse:
            tf.get_variable_scope().reuse_variables()
        net = input
        if input_norm:
            net = (net - input_norm[0]) / input_norm[1]
        last_layer_act = None
        for layer_config in network_config:
            if layer_config['TYPE'] == 'DENSE':
                if layer_config['B_INIT_VALUE'] is None:
                    b_init = None
                else:
                    b_init = tf.constant_initializer(value=layer_config['B_INIT_VALUE'])
                l1_norm = layer_config['L1_NORM'] if 'L1_NORM' in layer_config else 0.0
                l2_norm = layer_config['L2_NORM'] if 'L2_NORM' in layer_config else 0.0
                net = tf.layers.dense(inputs=net,
                                      units=layer_config['N_UNITS'],
                                      activation=MLPCreator.act_dict[layer_config['ACT']],
                                      use_bias=b_init is not None,
                                      kernel_initializer=contrib_W_init(),
                                      kernel_regularizer=tf_contrib.layers.l1_l2_regularizer(l1_norm, l2_norm),
                                      bias_regularizer=tf_contrib.layers.l1_l2_regularizer(l1_norm, l2_norm),
                                      bias_initializer=b_init,
                                      name=net_name + '_' + layer_config['NAME'],
                                      reuse=reuse
                                      )
                last_layer_act = layer_config['ACT']
        if output_norm:
            net = (net * output_norm[0]) + output_norm[1]
        if output_high is not None and output_low is not None:
            if last_layer_act not in ("IDENTITY", 'LINEAR'):
                raise ValueError(
                    'Please set the last layer activation as IDENTITY/LINEAR to use output scale, TANH will added to it as default')
            net = tf.tanh(net)
            net = (net + 1.0) / 2.0 * (output_high - output_low) + output_low
        # todo the collection may contain extra variable that is instanced by others but have same name scope
        net_all_params = get_tf_collection_var_list(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope=tf.get_variable_scope().name)
        if tf_var_scope_context is not None:
            tf_var_scope_context.__exit__(type_arg=None, value_arg=None, traceback_arg=None)
        assert tf.get_variable_scope().name == pre_var_scope_name
        return net, net, net_all_params
