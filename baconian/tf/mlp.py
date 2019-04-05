from typeguard import typechecked
from baconian.tf.util import MLPCreator
import tensorflow as tf
import numpy as np
from baconian.tf.tf_parameters import ParametersWithTensorflowVariable


class MLP(object):

    @typechecked
    def __init__(self,
                 input_ph: tf.Tensor,
                 name_scope: str,
                 net_name: str,
                 reuse,
                 mlp_config: list,
                 input_norm: np.ndarray = None,
                 output_norm: np.ndarray = None,
                 output_low: np.ndarray = None,
                 output_high: np.ndarray = None,
                 ):
        self.input_ph = input_ph
        self.name_scope = name_scope
        self.mlp_config = mlp_config
        self.mlp_net_name = net_name
        self.net, self.output, self.var_list = MLPCreator.create_network_with_tf_layers(input=input_ph,
                                                                                        reuse=reuse,
                                                                                        network_config=mlp_config,
                                                                                        tf_var_scope=name_scope,
                                                                                        net_name=net_name,
                                                                                        input_norm=input_norm,
                                                                                        output_high=output_high,
                                                                                        output_low=output_low,
                                                                                        output_norm=output_norm)
        for var in self.var_list:
            assert name_scope in var.name
        self._parameters = ParametersWithTensorflowVariable(tf_var_list=self.var_list,
                                                            name='parameters_{}'.format(self.mlp_net_name),
                                                            rest_parameters=dict())

    def forward(self, input: np.ndarray, sess=tf.get_default_session()) -> np.ndarray:
        feed_dict = {
            self.input_ph: input,
            **self._parameters.return_tf_parameter_feed_dict()
        }
        res = sess.run(self.output,
                       feed_dict=feed_dict)
        return np.squeeze(res)

    def copy_from(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        self._parameters.copy_from(source_parameter=obj._parameters)
        return True

    def init(self, source_obj=None):
        self._parameters.init()
        if source_obj:
            self.copy_from(obj=source_obj)
