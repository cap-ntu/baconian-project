from typeguard import typechecked
from src.tf.util import MLPCreator
import tensorflow as tf
import numpy as np


class MLP(object):
    """
    This class shall be DEPRECATED
    """

    @typechecked
    def __init__(self, config_list: list, input_ph: tf.Tensor, name_scope: str, mlp_net_name: str):
        self.input_ph = input_ph
        self.name_scope = name_scope
        self.config_list = config_list
        self.mlp_net_name = mlp_net_name

        self.net, self.output, self.var_list = MLPCreator.create_network(input=input_ph,
                                                                         network_config=config_list,
                                                                         tf_var_scope=name_scope,
                                                                         net_name=mlp_net_name)

    def forward(self, input: np.ndarray, sess=tf.get_default_session()) -> np.ndarray:
        res = sess.run(self.output,
                       feed_dict={self.input_ph: input})
        return np.squeeze(res)

    def copy_from(self, source_mlp):
        raise NotImplementedError

    # def init(self):
    #     tf_sess = tf.get_default_session()
    #     tf_sess.run(self.var_init_op)
    #     return True
