from baconian.algo.policy.policy import DeterministicPolicy
from baconian.core.core import EnvSpec
import overrides
import tensorflow as tf
from baconian.tf.mlp import MLP
from baconian.tf.tf_parameters import ParametersWithTensorflowVariable
from baconian.common.special import *
from baconian.algo.utils import _get_copy_arg_with_tf_reuse
from baconian.algo.misc.placeholder_input import PlaceholderInput


class DeterministicMLPPolicy(DeterministicPolicy, PlaceholderInput):

    def __init__(self, env_spec: EnvSpec,
                 name_scope: str,
                 name: str, mlp_config: list,
                 input_norm: np.ndarray = None,
                 output_norm: np.ndarray = None,
                 output_low: np.ndarray = None,
                 output_high: np.ndarray = None,
                 reuse=False):
        DeterministicPolicy.__init__(self, env_spec=env_spec, name=name, parameters=None)
        obs_dim = env_spec.flat_obs_dim
        action_dim = env_spec.flat_action_dim
        assert action_dim == mlp_config[-1]['N_UNITS']

        with tf.variable_scope(name_scope):
            state_input = tf.placeholder(shape=[None, obs_dim], dtype=tf.float32, name='state_ph')

        mlp_kwargs = dict(
            reuse=reuse,
            input_norm=input_norm,
            output_norm=output_norm,
            output_low=output_low,
            output_high=output_high,
            mlp_config=mlp_config,
            name_scope=name_scope
        )

        mlp_net = MLP(input_ph=state_input,
                      **mlp_kwargs,
                      net_name='deterministic_mlp_policy')

        PlaceholderInput.__init__(self, parameters=None)
        self.parameters = ParametersWithTensorflowVariable(tf_var_list=mlp_net.var_list,
                                                           rest_parameters=mlp_kwargs,
                                                           name='deterministic_mlp_policy_tf_param')
        self.state_input = state_input
        self.mlp_net = mlp_net
        self.action_tensor = mlp_net.output
        self.mlp_config = mlp_config
        self.mlp_config = mlp_config
        self.input_norm = input_norm
        self.output_norm = output_norm
        self.output_low = output_low
        self.output_high = output_high
        self.name_scope = name_scope

    @overrides.overrides
    def forward(self, obs: (np.ndarray, list), sess=None, feed_dict=None, **kwargs):
        obs = make_batch(obs, original_shape=self.env_spec.obs_shape)
        feed_dict = {} if feed_dict is None else feed_dict
        feed_dict = {
            **feed_dict,
            self.state_input: obs,
            **self.parameters.return_tf_parameter_feed_dict()
        }
        sess = sess if sess else tf.get_default_session()
        res = sess.run(self.action_tensor, feed_dict=feed_dict)
        res = np.clip(res, a_min=self.env_spec.action_space.low, a_max=self.env_spec.action_space.high)
        return res

    @overrides.overrides
    def copy_from(self, obj) -> bool:
        return PlaceholderInput.copy_from(self, obj)

    def make_copy(self, *args, **kwargs):
        kwargs = _get_copy_arg_with_tf_reuse(obj=self, kwargs=kwargs)

        copy_mlp_policy = DeterministicMLPPolicy(env_spec=self.env_spec,
                                                 input_norm=self.input_norm,
                                                 output_norm=self.output_norm,
                                                 output_low=self.output_low,
                                                 output_high=self.output_high,
                                                 mlp_config=self.mlp_config,
                                                 **kwargs)
        return copy_mlp_policy

    def save(self, *args, **kwargs):
        return PlaceholderInput.save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        return PlaceholderInput.load(self, *args, **kwargs)
