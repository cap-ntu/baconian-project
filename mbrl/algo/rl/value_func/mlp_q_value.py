from mbrl.algo.rl.value_func.value_func import PlaceholderInputValueFunction
import typeguard as tg
from mbrl.envs.env_spec import EnvSpec
import overrides
import numpy as np
import tensorflow as tf
from typeguard import typechecked
from mbrl.tf.tf_parameters import TensorflowParameters
from mbrl.tf.mlp import MLP
from mbrl.common.special import *
from mbrl.core.util import init_func_arg_record_decorator


class MLPQValueFunction(PlaceholderInputValueFunction):
    """
    Multi Layer Q Value Function, based on Tensorflow, take the state and action as input,
    return the Q value for all action/ input action.
    """

    @init_func_arg_record_decorator()
    @tg.typechecked
    def __init__(self,
                 env_spec: EnvSpec,
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
        with tf.name_scope(name_scope):
            self.state_input = state_input if state_input is not None else tf.placeholder(
                shape=[None, env_spec.flat_obs_dim],
                dtype=tf.float32,
                name='state_ph')
            self.action_input = action_input if action_input is not None else tf.placeholder(
                shape=[None, env_spec.flat_action_dim],
                dtype=tf.float32,
                name='action_ph')
        self.name_scope = name_scope
        self.mlp_config = mlp_config
        self.input_norm = input_norm
        self.output_norm = output_norm
        self.output_low = output_low
        self.output_high = output_high

        with tf.variable_scope(self.name_scope):
            self.mlp_input_ph = tf.concat([self.state_input, self.action_input], axis=1, name='state_action_input')

        self.mlp_net = MLP(input_ph=self.mlp_input_ph,
                           reuse=reuse,
                           mlp_config=mlp_config,
                           input_norm=input_norm,
                           output_norm=output_norm,
                           output_high=output_high,
                           output_low=output_low,
                           name_scope=name_scope,
                           net_name='mlp')
        self.q_tensor = self.mlp_net.output
        parameters = TensorflowParameters(tf_var_list=self.mlp_net.var_list,
                                          rest_parameters=dict(),
                                          name='mlp_q_value_function_tf_param',
                                          auto_init=False)
        super(MLPQValueFunction, self).__init__(env_spec=env_spec,
                                                parameters=parameters,
                                                input=self.mlp_input_ph)

    @overrides.overrides
    def copy(self, obj: PlaceholderInputValueFunction) -> bool:
        assert super().copy(obj) is True
        self.parameters.copy_from(source_parameter=obj.parameters)
        return True

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
            self.copy(obj=source_obj)

    def make_copy(self, *args, **kwargs):
        if 'reuse' in kwargs:
            if kwargs['reuse'] is True:
                if 'name_scope' in kwargs and kwargs['name_scope'] != self.name_scope:
                    raise ValueError('If reuse, the name scope should be same, but is different now: {} and {}'.format(
                        kwargs['name_scope'], self.name_scope))
                else:
                    kwargs.update(name_scope=self.name_scope)
            else:
                if 'name_scope' in kwargs and kwargs['name_scope'] == self.name_scope:
                    raise ValueError(
                        'If not reuse, the name scope should be different, but is same now: {} and {}'.format(
                            kwargs['name_scope'], self.name_scope))

        copy_mlp_q_value = MLPQValueFunction(env_spec=self.env_spec,
                                             input_norm=self.input_norm,
                                             output_norm=self.output_norm,
                                             output_low=self.output_low,
                                             output_high=self.output_high,
                                             mlp_config=self.mlp_config,
                                             **kwargs)
        return copy_mlp_q_value
