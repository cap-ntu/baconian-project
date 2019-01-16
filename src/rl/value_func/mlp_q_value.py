# Date: 11/16/18
# Author: Luke
# Project: ModelBasedRLFramework
from src.rl.value_func.value_func import ValueFunction
import typeguard as tg
from gym.core import Space
from src.envs.env_spec import EnvSpec
import overrides
import numpy as np
import tensorflow as tf
from src.tf.mlp import MLP
from typeguard import typechecked
from src.tf.tf_parameters import TensorflowParameters
from src.tf.util import MLPCreator


class MLPQValueFunction(ValueFunction):
    """
    Multi Layer Q Value Function, based on Tensorflow, take the state and action as input,
    return the Q value for all action/ input action.
    """

    @tg.typechecked
    def __init__(self,
                 env_spec: EnvSpec,
                 name_scope: str,
                 input_norm: bool,
                 output_norm: bool,
                 output_low: (list, np.ndarray, None),
                 output_high: (list, np.ndarray, None),
                 mlp_config: list):
        self.name_scope = name_scope
        self.mlp_config = mlp_config
        with tf.variable_scope(self.name_scope):
            self.state_ph = tf.placeholder(shape=[None, env_spec.flat_obs_dim], dtype=tf.float32, name='state_ph')
            self.action_ph = tf.placeholder(shape=[None, env_spec.flat_action_dim], dtype=tf.float32, name='action_ph')
            self.mlp_input = tf.concat([self.state_ph, self.action_ph], axis=1, name='state_action_input')

        self.mlp_net, self.q_tensor, var_list = MLPCreator.create_network(input=self.mlp_input,
                                                                          network_config=self.mlp_config,
                                                                          input_norm=input_norm,
                                                                          output_norm=output_norm,
                                                                          output_high=output_high,
                                                                          output_low=output_low,
                                                                          tf_var_scope=name_scope,
                                                                          net_name='mlq_q_value_function')
        parameters = TensorflowParameters(tf_var_list=var_list,
                                          name='mlp_q_value_function_tf_param',
                                          auto_init=False)

        super(MLPQValueFunction, self).__init__(env_spec=env_spec, parameters=parameters)

    @overrides.overrides
    def copy(self, obj: ValueFunction) -> bool:
        assert super().copy(obj) is True
        self.parameters.copy_from(source_parameter=obj.parameters)
        return True

    @typechecked
    @overrides.overrides
    def forward(self, obs: (np.ndarray, list), action: (np.ndarray, list), sess=tf.get_default_session(), *args,
                **kwargs):
        q = sess.run(self.q_tensor,
                     feed_dict={self.state_ph: obs,
                                self.action_ph: action})
        return q

    def init(self, source_obj=None):
        self.parameters.init()
        if source_obj:
            self.copy(obj=source_obj)
