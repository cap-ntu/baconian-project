import os

import tensorflow as tf
# import tensorlayer as tl
from tensorflow.contrib.layers import variance_scaling_initializer as contrib_W_init
from typeguard import typechecked


def create_new_tf_session(cuda_device: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.__enter__()
    assert tf.get_default_session()
    return sess


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

    # @staticmethod
    # @typechecked
    # def create_network(input: tf.Tensor, network_config: list, net_name='', tf_var_scope=None, input_norm=None,
    #                    output_norm=None,
    #                    reuse=False,
    #                    output_low=None, output_high=None):
    #     """
    #     Create a MLP network with a input tensor
    #     warning: this will create a input net which will cut the gradients from the input tensor and its
    #     previous op
    #     :param input:
    #     :param network_config:
    #     :param net_name:
    #     :param tf_var_scope:
    #     :param input_norm:
    #     :param output_norm:
    #     :param output_low:
    #     :param output_high:
    #     :return:
    #     """
    #     if tf_var_scope is not None:
    #         # TODO test: this context need to be checked
    #         tf_var_scope = tf.variable_scope(tf_var_scope)
    #         tf_var_scope.__enter__()
    #         if reuse:
    #             # todo debug reuse
    #             tf.get_variable_scope().reuse_variables()
    #     net = tl.layers.InputLayer(inputs=input,
    #                                name=net_name + '_INPUT')
    #
    #     if input_norm:
    #         net = tl.layers.LambdaLayer(prev_layer=net,
    #                                     name="{}_INPUT_NORM".format(net_name),
    #                                     fn=lambda x: (x - input_norm[0]) / input_norm[1])
    #     last_layer_act = None
    #     for layer_config in network_config:
    #         if layer_config['TYPE'] == 'DENSE':
    #             if layer_config['B_INIT_VALUE'] == 'None':
    #                 b_init = None
    #             else:
    #                 b_init = tf.constant_initializer(value=layer_config['B_INIT_VALUE'])
    #
    #             net = tl.layers.DenseLayer(prev_layer=net,
    #                                        n_units=layer_config['N_UNITS'],
    #                                        act=MLPCreator.act_dict[layer_config['ACT']],
    #                                        name=net_name + '_' + layer_config['NAME'],
    #                                        W_init=contrib_W_init(),
    #                                        b_init=b_init
    #                                        )
    #             last_layer_act = layer_config['ACT']
    #     if output_norm:
    #         net = tl.layers.LambdaLayer(prev_layer=net,
    #                                     fn=lambda x: (x * output_norm[0]) + output_norm[1],
    #                                     name=net_name + '_NORM')
    #     if output_high is not None and output_low is not None:
    #         if last_layer_act not in ("IDENTITY", 'LINEAR'):
    #             raise ValueError('Please set the last layer activation as IDENTITY/LINEAR to use output scale')
    #         net = tl.layers.LambdaLayer(prev_layer=net,
    #                                     fn=lambda x: tf.nn.tanh(x),
    #                                     name=net_name + '_TANH')
    #         net = tl.layers.LambdaLayer(prev_layer=net,
    #                                     fn=lambda x: (x + 1.0) / 2.0 * (output_high - output_low) + output_low,
    #                                     name=net_name + '_NORM_AFTER_TANH')
    #     if tf_var_scope is not None:
    #         # TODO test: this context need to be checked
    #         tf_var_scope.__exit__(type_arg=None, value_arg=None, traceback_arg=None)
    #     return net, net.outputs, net.all_params

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
        # TODO test: this context need to be checked
        tf_var_scope_context = tf.variable_scope(tf_var_scope)
        tf_var_scope_context.__enter__()
        if reuse:
            # todo debug reuse
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
        net_all_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
        if tf_var_scope_context is not None:
            # TODO test: this context need to be checked
            tf_var_scope_context.__exit__(type_arg=None, value_arg=None, traceback_arg=None)
        return net, net, net_all_params
