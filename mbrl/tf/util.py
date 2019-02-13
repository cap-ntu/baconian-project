import os

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer as contrib_W_init
from typeguard import typechecked

__all__ = ['get_tf_collection_var_list', 'MLPCreator']


def get_tf_collection_var_list(scope, key=tf.GraphKeys.GLOBAL_VARIABLES):
    var_list = tf.get_collection(key, scope=scope)
    return sorted(list(set(var_list)), key=lambda x: x.name)


def create_new_tf_session(cuda_device: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.__enter__()
    assert tf.get_default_session()
    return sess


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
    @typechecked
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
                if layer_config['B_INIT_VALUE'] == 'None':
                    b_init = None
                else:
                    b_init = tf.constant_initializer(value=layer_config['B_INIT_VALUE'])
                net = tf.layers.dense(inputs=net,
                                      units=layer_config['N_UNITS'],
                                      activation=MLPCreator.act_dict[layer_config['ACT']],
                                      use_bias=b_init is not None,
                                      kernel_initializer=contrib_W_init(),
                                      bias_initializer=b_init,
                                      name=net_name + '_' + layer_config['NAME'],
                                      reuse=reuse
                                      )
                last_layer_act = layer_config['ACT']
        if output_norm:
            net = (net * output_norm[0]) + output_norm[1]
        if output_high is not None and output_low is not None:
            if last_layer_act not in ("IDENTITY", 'LINEAR'):
                raise ValueError('Please set the last layer activation as IDENTITY/LINEAR to use output scale')
            net = tf.tanh(net)
            net = (net + 1.0) / 2.0 * (output_high - output_low) + output_low
        # todo bugs here: the collection may contain extra variable that is instanced by others but have same name scope
        net_all_params = get_tf_collection_var_list(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope=tf.get_variable_scope().name)
        if tf_var_scope_context is not None:
            tf_var_scope_context.__exit__(type_arg=None, value_arg=None, traceback_arg=None)
        assert tf.get_variable_scope().name == pre_var_scope_name
        return net, net, net_all_params
