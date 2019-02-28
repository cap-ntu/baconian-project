import typeguard as tg
from mobrl.core.core import EnvSpec
import overrides
import numpy as np
import tensorflow as tf
from typeguard import typechecked
from mobrl.tf.tf_parameters import TensorflowParameters
from mobrl.tf.mlp import MLP
from mobrl.common.special import *
from mobrl.core.util import init_func_arg_record_decorator
from mobrl.algo.rl.utils import _get_copy_arg_with_tf_reuse
from mobrl.algo.placeholder_input import PlaceholderInput
from mobrl.algo.rl.value_func.value_func import ValueFunction


class MLPQValueFunction(ValueFunction, PlaceholderInput):
    """
    Multi Layer Q Value Function, based on Tensorflow, take the state and action as input,
    return the Q value for all action/ input action.
    """

    @init_func_arg_record_decorator()
    @tg.typechecked
    def __init__(self,
                 env_spec: EnvSpec,
                 name: str,
                 name_scope: str,
                 mlp_config: list,
                 state_input: tf.Tensor = None,
                 action_input: tf.Tensor = None,
                 reuse=False,
                 input_norm: np.ndarray = None,
                 output_norm: np.ndarray = None,
                 output_low: np.ndarray = None,
                 output_high: np.ndarray = None,
                 ):
        # todo have to set name first because of tf parameters late initialization
        self._name = name
        with tf.name_scope(self.name):
            self.state_input = state_input if state_input is not None else tf.placeholder(
                shape=[None, env_spec.flat_obs_dim],
                dtype=tf.float32,
                name='state_ph')
            self.action_input = action_input if action_input is not None else tf.placeholder(
                shape=[None, env_spec.flat_action_dim],
                dtype=tf.float32,
                name='action_ph')
        self.mlp_config = mlp_config
        self.input_norm = input_norm
        self.output_norm = output_norm
        self.output_low = output_low
        self.output_high = output_high
        self.name_scope = name_scope
        with tf.variable_scope(self.name_scope):
            self.mlp_input_ph = tf.concat([self.state_input, self.action_input], axis=1, name='state_action_input')

        self.mlp_net = MLP(input_ph=self.mlp_input_ph,
                           reuse=reuse,
                           mlp_config=mlp_config,
                           input_norm=input_norm,
                           output_norm=output_norm,
                           output_high=output_high,
                           output_low=output_low,
                           name_scope=self.name_scope,
                           net_name='mlp')
        self.q_tensor = self.mlp_net.output
        parameters = TensorflowParameters(tf_var_list=self.mlp_net.var_list,
                                          rest_parameters=dict(),
                                          default_save_type='tf',
                                          name='{}_tf_param'.format(self.name),
                                          auto_init=False)
        ValueFunction.__init__(self,
                               env_spec=env_spec,
                               name=name,
                               parameters=parameters)
        PlaceholderInput.__init__(self, parameters=parameters, inputs=self.mlp_input_ph)

    @overrides.overrides
    def copy_from(self, obj: PlaceholderInput) -> bool:
        return PlaceholderInput.copy_from(self, obj)

    @typechecked
    @overrides.overrides
    def forward(self, obs: (np.ndarray, list), action: (np.ndarray, list), sess=None,
                feed_dict=None, *args,
                **kwargs):
        sess = sess if sess else tf.get_default_session()
        obs = make_batch(obs, original_shape=self.env_spec.obs_shape)
        action = make_batch(action, original_shape=self.env_spec.action_shape)
        feed_dict = {
            self.state_input: obs,
            self.action_input: action,
            **self.parameters.return_tf_parameter_feed_dict()
        } if feed_dict is None else {
            **feed_dict,
            **self.parameters.return_tf_parameter_feed_dict()
        }
        q = sess.run(self.q_tensor,
                     feed_dict=feed_dict)
        return q

    def init(self, source_obj=None):
        self.parameters.init()
        if source_obj:
            self.copy_from(obj=source_obj)

    def make_copy(self, *args, **kwargs):
        kwargs = _get_copy_arg_with_tf_reuse(obj=self, kwargs=kwargs)

        copy_mlp_q_value = MLPQValueFunction(env_spec=self.env_spec,
                                             input_norm=self.input_norm,
                                             output_norm=self.output_norm,
                                             output_low=self.output_low,
                                             output_high=self.output_high,
                                             mlp_config=self.mlp_config,
                                             **kwargs)
        return copy_mlp_q_value

    def save(self, *args, **kwargs):
        return PlaceholderInput.save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        return PlaceholderInput.load(self, *args, **kwargs)