import typeguard as tg
from baconian.core.core import EnvSpec
import overrides
import tensorflow as tf
from baconian.tf.tf_parameters import ParametersWithTensorflowVariable
from baconian.tf.mlp import MLP
from baconian.common.special import *
from baconian.core.util import init_func_arg_record_decorator
from baconian.algo.utils import _get_copy_arg_with_tf_reuse
from baconian.algo.misc.placeholder_input import PlaceholderInput
from baconian.algo.value_func import QValueFunction


class MLPQValueFunction(QValueFunction, PlaceholderInput):
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
        with tf.name_scope(name_scope):
            state_input = state_input if state_input is not None else tf.placeholder(
                shape=[None, env_spec.flat_obs_dim],
                dtype=tf.float32,
                name='state_ph')
            action_input = action_input if action_input is not None else tf.placeholder(
                shape=[None, env_spec.flat_action_dim],
                dtype=tf.float32,
                name='action_ph')
        with tf.variable_scope(name_scope):
            mlp_input_ph = tf.concat([state_input, action_input], axis=1, name='state_action_input')
        mlp_net_kwargs = dict(
            reuse=reuse,
            mlp_config=mlp_config,
            input_norm=input_norm,
            output_norm=output_norm,
            output_high=output_high,
            output_low=output_low,
            name_scope=name_scope,
        )
        mlp_net = MLP(input_ph=mlp_input_ph,
                      net_name=name_scope,
                      **mlp_net_kwargs)
        parameters = ParametersWithTensorflowVariable(tf_var_list=mlp_net.var_list,
                                                      rest_parameters=dict(
                                                          **mlp_net_kwargs,
                                                          name=name
                                                      ),
                                                      default_save_type='tf',
                                                      name='{}_tf_param'.format(name))
        QValueFunction.__init__(self,
                                env_spec=env_spec,
                                name=name,
                                action_input=action_input,
                                state_input=state_input,
                                parameters=None)
        PlaceholderInput.__init__(self, parameters=parameters)

        self.mlp_config = mlp_config
        self.input_norm = input_norm
        self.output_norm = output_norm
        self.output_low = output_low
        self.output_high = output_high
        self.name_scope = name_scope
        self.mlp_input_ph = mlp_input_ph
        self.mlp_net = mlp_net
        self.q_tensor = self.mlp_net.output

    @overrides.overrides
    def copy_from(self, obj: PlaceholderInput) -> bool:
        return PlaceholderInput.copy_from(self, obj)

    @overrides.overrides
    def forward(self, obs: (np.ndarray, list), action: (np.ndarray, list), sess=None,
                feed_dict=None, *args,
                **kwargs):
        sess = sess if sess else tf.get_default_session()
        obs = make_batch(obs, original_shape=self.env_spec.obs_shape)
        action = make_batch(action, original_shape=[self.env_spec.flat_action_dim])
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
