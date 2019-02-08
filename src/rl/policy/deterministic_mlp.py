from src.envs.env_spec import EnvSpec
from src.rl.policy.policy import DeterministicPolicy
from typeguard import typechecked
from gym.core import Space
from src.envs.env_spec import EnvSpec
import overrides
import numpy as np
import tensorflow as tf
from src.tf.mlp import MLP
from src.tf.tf_parameters import TensorflowParameters
from src.common.misc.misc import *
from src.common.misc.special import *


class DeterministicMLPPolicy(DeterministicPolicy):

    def __init__(self, env_spec: EnvSpec,
                 name_scope: str, mlp_config: list,
                 input_norm: np.ndarray = None,
                 output_norm: np.ndarray = None,
                 output_low: np.ndarray = None,
                 output_high: np.ndarray = None,
                 reuse=False):
        super(DeterministicMLPPolicy, self).__init__(env_spec, parameters=None)
        obs_dim = env_spec.flat_obs_dim
        action_dim = env_spec.flat_action_dim
        # todo check the key here
        assert action_dim == mlp_config[-1]['N_UNITS']
        self.name_scope = name_scope
        self.mlp_config = mlp_config
        self.input_norm = input_norm
        self.output_norm = output_norm
        self.output_low = output_low
        self.output_high = output_high

        with tf.variable_scope(name_scope):
            self.state_input = tf.placeholder(shape=[None, obs_dim], dtype=tf.float32, name='state_ph')
        self.mlp_net = MLP(input_ph=self.state_input,
                           reuse=reuse,
                           input_norm=input_norm,
                           output_norm=output_norm,
                           output_low=output_low,
                           output_high=output_high,
                           net_name='deterministic_mlp_policy',
                           mlp_config=mlp_config,
                           name_scope=name_scope)

        self.action_tensor = self.mlp_net.output
        self.mlp_config = mlp_config
        self.parameters = TensorflowParameters(tf_var_list=self.mlp_net.var_list,
                                               rest_parameters=dict(),
                                               name='deterministic_mlp_policy_tf_param',
                                               auto_init=False)

    @typechecked
    @overrides.overrides
    def forward(self, obs: (np.ndarray, list), sess=None, feed_dict=None, **kwargs):
        obs = make_batch(obs, original_shape=self.env_spec.obs_shape)
        feed_dict = {
            self.state_input: obs,
            **self.parameters.return_tf_parameter_feed_dict()
        } if feed_dict is None else {
            **feed_dict,
            **self.parameters.return_tf_parameter_feed_dict()
        }
        sess = sess if sess else tf.get_default_session()
        res = sess.run(self.action_tensor, feed_dict=feed_dict)
        # todo clip the action?
        res = np.clip(res, a_min=self.env_spec.action_space.low, a_max=self.env_spec.action_space.high)
        return res

    @overrides.overrides
    def copy(self, obj) -> bool:
        assert isinstance(obj, (type(self), MLP))
        self.mlp_net.copy(obj=obj.mlp_net if isinstance(obj, type(self)) else obj)
        return super().copy(obj)

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

        copy_mlp_policy = DeterministicMLPPolicy(env_spec=self.env_spec,
                                                 input_norm=self.input_norm,
                                                 output_norm=self.output_norm,
                                                 output_low=self.output_low,
                                                 output_high=self.output_high,
                                                 mlp_config=self.mlp_config,
                                                 **kwargs)
        return copy_mlp_policy

    def init(self, source_obj=None):
        self.parameters.init()
        if source_obj:
            self.copy(obj=source_obj)
