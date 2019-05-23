from baconian.algo.policy.policy import StochasticPolicy
from baconian.core.core import EnvSpec
import overrides
import tensorflow as tf
from baconian.tf.mlp import MLP
from baconian.tf.tf_parameters import ParametersWithTensorflowVariable
from baconian.common.special import *
import tensorflow_probability as tfp
from baconian.tf.util import *
from baconian.algo.utils import _get_copy_arg_with_tf_reuse
from baconian.algo.misc.placeholder_input import PlaceholderInput

"""
logvar and logvar_speed is referred from https://github.com/pat-coady/trpo

"""


class NormalDistributionMLPPolicy(StochasticPolicy, PlaceholderInput):

    def __init__(self, env_spec: EnvSpec,
                 name: str,
                 name_scope: str,
                 mlp_config: list,
                 input_norm: np.ndarray = None,
                 output_norm: np.ndarray = None,
                 output_low: np.ndarray = None,
                 output_high: np.ndarray = None,
                 reuse=False,
                 distribution_tensors_tuple: tuple = None
                 ):
        StochasticPolicy.__init__(self, env_spec=env_spec, name=name, parameters=None)
        obs_dim = env_spec.flat_obs_dim
        action_dim = env_spec.flat_action_dim
        assert action_dim == mlp_config[-1]['N_UNITS']
        self.mlp_config = mlp_config
        self.input_norm = input_norm
        self.output_norm = output_norm
        self.output_low = output_low
        self.output_high = output_high
        self.mlp_config = mlp_config
        self.name_scope = name_scope

        mlp_kwargs = dict(
            reuse=reuse,
            input_norm=input_norm,
            output_norm=output_norm,
            output_low=output_low,
            output_high=output_high,
            mlp_config=mlp_config,
            name_scope=name_scope
        )
        ph_inputs = []
        if distribution_tensors_tuple is not None:
            self.mean_output = distribution_tensors_tuple[0][0]
            self.logvar_output = distribution_tensors_tuple[1][0]
            assert list(self.mean_output.shape)[-1] == action_dim
            assert list(self.logvar_output.shape)[-1] == action_dim
            self.mlp_net = None
        else:
            with tf.variable_scope(self.name_scope):
                self.state_input = tf.placeholder(shape=[None, obs_dim], dtype=tf.float32, name='state_ph')
                ph_inputs.append(self.state_input)
            self.mlp_net = MLP(input_ph=self.state_input,
                               net_name='normal_distribution_mlp_policy',
                               **mlp_kwargs)
            self.mean_output = self.mlp_net.output
            with tf.variable_scope(name_scope, reuse=reuse):
                with tf.variable_scope('norm_dist', reuse=reuse):
                    logvar_speed = (10 * self.mlp_config[-2]['N_UNITS']) // 48
                    logvar_output = tf.get_variable(name='normal_distribution_variance',
                                                    shape=[logvar_speed, self.mlp_config[-1]['N_UNITS']],
                                                    dtype=tf.float32)
                    # self.logvar_output = tf.reduce_sum(logvar_output, axis=0) + self.parameters('log_var_init')
                    self.logvar_output = tf.reduce_sum(logvar_output, axis=0)
        with tf.variable_scope(name_scope, reuse=reuse):
            self.action_input = tf.placeholder(shape=[None, action_dim], dtype=tf.float32, name='action_ph')
            ph_inputs.append(self.action_input)
            with tf.variable_scope('norm_dist', reuse=reuse):
                self.stddev_output = tf.exp(self.logvar_output / 2.0, name='std_dev')
                self.var_output = tf.exp(self.logvar_output, name='variance')
                self.action_distribution = tfp.distributions.MultivariateNormalDiag(loc=self.mean_output,
                                                                                    scale_diag=self.stddev_output,
                                                                                    name='mlp_normal_distribution')
                self.action_output = self.action_distribution.sample()
        self.dist_info_tensor_op_dict = {
            # todo support more in future
            'prob': self.action_distribution.prob,
            'log_prob': self.action_distribution.log_prob,
            'entropy': self.action_distribution.entropy,
            'kl': self.kl
        }
        var_list = get_tf_collection_var_list(scope='{}/norm_dist'.format(name_scope))
        if self.mlp_net:
            var_list += self.mlp_net.var_list

        self.parameters = ParametersWithTensorflowVariable(tf_var_list=sorted(list(set(var_list)),
                                                                              key=lambda x: x.name),
                                                           rest_parameters=dict(
                                                               state_input=self.state_input,
                                                               action_input=self.action_input,
                                                               **mlp_kwargs
                                                           ),
                                                           name='normal_distribution_mlp_tf_param')
        PlaceholderInput.__init__(self, parameters=self.parameters, inputs=tuple(ph_inputs))

    @typechecked
    @overrides.overrides
    def forward(self, obs: (np.ndarray, list), sess=None, feed_dict=None, **kwargs):
        obs = make_batch(obs, original_shape=self.env_spec.obs_shape)
        feed_dict = feed_dict if feed_dict is not None else dict()
        feed_dict = {
            **feed_dict,
            self.state_input: obs,
            **self.parameters.return_tf_parameter_feed_dict()
        }
        sess = sess if sess else tf.get_default_session()
        res = sess.run(self.action_output, feed_dict=feed_dict)
        res = np.clip(res, a_min=self.env_spec.action_space.low, a_max=self.env_spec.action_space.high)
        return res

    @overrides.overrides
    def copy_from(self, obj) -> bool:
        return PlaceholderInput.copy_from(self, obj)

    def make_copy(self, **kwargs):
        kwargs = _get_copy_arg_with_tf_reuse(obj=self, kwargs=kwargs)
        copy_mlp_policy = NormalDistributionMLPPolicy(env_spec=self.env_spec,
                                                      input_norm=self.input_norm,
                                                      output_norm=self.output_norm,
                                                      output_low=self.output_low,
                                                      output_high=self.output_high,
                                                      mlp_config=self.mlp_config,
                                                      **kwargs)
        return copy_mlp_policy

    def compute_dist_info(self, name, sess=None, **kwargs) -> np.ndarray:
        assert name in ['log_prob', 'prob', 'entropy', 'kl']
        sess = sess if sess else tf.get_default_session()
        if name in ['log_prob', 'prob']:
            if 'value' not in kwargs:
                raise ValueError('To compute {}, pass the parameter with key {} and type {} in'.format(name, 'value',
                                                                                                       np.ndarray.__name__))
            assert isinstance(kwargs['value'], np.ndarray)
        if name == 'kl':
            if 'other' not in kwargs:
                raise ValueError('To compute {}, pass the parameter with key {} and type {} in'.format(name, 'other',
                                                                                                       type(
                                                                                                           self).__name__))
            assert isinstance(kwargs['other'], type(self))
        if 'feed_dict' in kwargs:
            feed_dict = kwargs['feed_dict'] if 'feed_dict' in kwargs else None
            kwargs.pop('feed_dict')
        else:
            feed_dict = None

        return sess.run(self.dist_info_tensor_op_dict[name](**kwargs), feed_dict=feed_dict)

    def kl(self, other, *args, **kwargs) -> tf.Tensor:
        if not isinstance(other.action_distribution, tfp.distributions.Distribution):
            raise TypeError()
        return self.action_distribution.kl_divergence(other.action_distribution)

    def log_prob(self, *args, **kwargs) -> tf.Tensor:
        return self.dist_info_tensor_op_dict['log_prob'](value=self.action_input)

    def prob(self, *args, **kwargs) -> tf.Tensor:
        return self.dist_info_tensor_op_dict['prob'](value=self.action_input)

    def entropy(self, *args, **kwargs) -> tf.Tensor:
        return self.dist_info_tensor_op_dict['entropy']()

    def get_dist_info(self) -> tuple:
        res = (
            dict(shape=tuple(self.mean_output.shape.as_list()),
                 name='mean_output',
                 obj=self.mean_output,
                 dtype=self.mean_output.dtype),
            dict(shape=tuple(self.logvar_output.shape.as_list()),
                 name='logvar_output',
                 obj=self.logvar_output,
                 dtype=self.logvar_output.dtype)
        )
        for re in res:
            attr = getattr(self, re['name'])
            if id(attr) != id(re['obj']):
                raise ValueError('key name {} should be same as the obj {} name'.format(re['name'], re['obj']))
        return res

    def save(self, *args, **kwargs):
        return PlaceholderInput.save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        return PlaceholderInput.load(self, *args, **kwargs)


if __name__ == '__main__':
    from baconian.test.tests.test_rl.test_policy.test_mlp_norm_policy import TestNormalDistMLPPolicy
    import unittest

    unittest.TestLoader().loadTestsFromTestCase(TestNormalDistMLPPolicy)
    unittest.main()
