from src.envs.env_spec import EnvSpec
from src.rl.policy.policy import StochasticPolicy
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
import tensorflow_probability as tfp


class NormalDistributionMLPPolicy(StochasticPolicy):

    def __init__(self, env_spec: EnvSpec,
                 name_scope: str, mlp_config: list,
                 input_norm: np.ndarray = None,
                 output_norm: np.ndarray = None,
                 output_low: np.ndarray = None,
                 output_high: np.ndarray = None,
                 reuse=False):
        super(NormalDistributionMLPPolicy, self).__init__(env_spec, parameters=None)
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
                           net_name='normal_distribution_mlp_policy',
                           mlp_config=mlp_config,
                           name_scope=name_scope)
        self.mean_tenor = self.mlp_net.output
        with tf.variable_scope(name_scope):
            with tf.variable_scope('norm_dist', reuse=reuse):
                # logvar and logvar_speed is referred from
                # https://github.com/pat-coady/trpo
                logvar_speed = (10 * self.mlp_config[-2]['N_UNITS']) // 48
                self.logvar_tensor = tf.get_variable(name='normal_distribution_variance',
                                                     shape=[logvar_speed, self.mlp_config[-1]['N_UNITS']],
                                                     dtype=tf.float32)
                self.stddev_tensor = tf.exp(self.logvar_tensor / 2.0)
                self.action_distribution = tfp.distributions.MultivariateNormalDiag(loc=self.mean_tenor,
                                                                                    scale_diag=self.stddev_tensor,
                                                                                    name='mlp_normal_distribution')
                self.action_tensor = self.action_distribution.sample()
        self.mlp_config = mlp_config
        var_list = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='{}/norm_dist'.format(name_scope))
        # todo use set to solve the duplicated tensor bug temporarily
        self.parameters = TensorflowParameters(tf_var_list=sorted(list(set(self.mlp_net.var_list + var_list)),
                                                                  key=lambda x: x.name),
                                               rest_parameters=dict(),
                                               name='normal_distribution_mlp_tf_param',
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
        assert isinstance(obj, type(self))
        self.mlp_net.copy(obj=obj.mlp_net)
        tf_sess = tf.get_default_session()
        tf_sess.run(tf.assign(self.logvar_tensor, obj.logvar_tensor))
        return super().copy(obj)

    def make_copy(self, **kwargs):
        if 'reuse' in kwargs:
            if kwargs['reuse'] is True:
                if 'name_scope' in kwargs and kwargs['name_scope'] != self.name_scope:
                    raise ValueError('If reuse, the name scope should be same instead of : {} and {}'.format(
                        kwargs['name_scope'], self.name_scope))
                else:
                    kwargs.update(name_scope=self.name_scope)
            else:
                if 'name_scope' in kwargs and kwargs['name_scope'] == self.name_scope:
                    raise ValueError(
                        'If not reuse, the name scope should be different instead of: {} and {}'.format(
                            kwargs['name_scope'], self.name_scope))

        copy_mlp_policy = NormalDistributionMLPPolicy(env_spec=self.env_spec,
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
