from typeguard import typechecked
from src.tf.util import MLPCreator
import tensorflow as tf
import numpy as np
from src.tf.tf_parameters import TensorflowParameters


class MLP(object):

    @typechecked
    def __init__(self,
                 input_ph: tf.Tensor,
                 name_scope: str,
                 net_name: str,
                 output_norm: bool,
                 input_norm: bool,
                 output_low: (list, np.ndarray, None),
                 output_high: (list, np.ndarray, None),
                 mlp_config: list):
        self.input_ph = input_ph
        self.name_scope = name_scope
        self.mlp_config = mlp_config
        self.mlp_net_name = net_name
        with tf.variable_scope(self.name_scope):
            self.net, self.output, self.var_list = MLPCreator.create_network(input=input_ph,
                                                                             network_config=mlp_config,
                                                                             tf_var_scope=name_scope,
                                                                             net_name=net_name,
                                                                             input_norm=input_norm,
                                                                             output_high=output_high,
                                                                             output_low=output_low,
                                                                             output_norm=output_norm)
            self.parameters = TensorflowParameters(tf_var_list=self.var_list,
                                                   name='parameters_{}'.format(self.mlp_net_name),
                                                   rest_parameters=dict(),
                                                   auto_init=False)

    def forward(self, input: np.ndarray, sess=tf.get_default_session()) -> np.ndarray:
        feed_dict = {
            self.input_ph: input,
            **self.parameters.return_tf_parameter_feed_dict()
        }
        res = sess.run(self.output,
                       feed_dict=feed_dict)
        return np.squeeze(res)

    def copy(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        self.parameters.copy_from(source_parameter=obj.parameters)
        return True

    def init(self, source_obj=None):
        self.parameters.init()
        if source_obj:
            self.copy(obj=source_obj)
