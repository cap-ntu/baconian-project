from mobrl.common.special import flatten_n
from mobrl.core.core import EnvSpec
from mobrl.algo.rl.model_based.models.dynamics_model import GlobalDynamicsModel, DerivableDynamics
import tensorflow as tf
from mobrl.tf.tf_parameters import TensorflowParameters
from mobrl.tf.mlp import MLP
import tensorflow.contrib as tf_contrib
from mobrl.common.sampler.sample_data import TransitionData
from typeguard import typechecked
import numpy as np
from mobrl.tf.util import *
from mobrl.algo.placeholder_input import PlaceholderInput


class ContinuousMLPGlobalDynamicsModel(GlobalDynamicsModel, DerivableDynamics, PlaceholderInput):
    def __init__(self, env_spec: EnvSpec,
                 name_scope: str,
                 name: str,
                 mlp_config: list,
                 learning_rate: float,
                 l1_norm_scale: float,
                 l2_norm_scale: float,
                 output_norm: np.ndarray = None,
                 input_norm: np.ndarray = None,
                 output_low: np.ndarray = None,
                 output_high: np.ndarray = None,
                 init_state=None):
        GlobalDynamicsModel.__init__(self, env_spec=env_spec, parameters=None,
                                     name=name,
                                     init_state=init_state)

        self.mlp_config = mlp_config
        self.name_scope = name_scope
        with tf.variable_scope(self.name_scope):
            self.state_input = tf.placeholder(shape=[None, env_spec.flat_obs_dim], dtype=tf.float32, name='state_ph')
            self.action_input = tf.placeholder(shape=[None, env_spec.flat_action_dim], dtype=tf.float32,
                                               name='action_ph')
            self.mlp_input_ph = tf.concat([self.state_input, self.action_input], axis=1, name='state_action_input')
            self.delta_state_label_ph = tf.placeholder(shape=[None, env_spec.flat_obs_dim], dtype=tf.float32,
                                                       name='delta_state_label_ph')
        self.mlp_net = MLP(input_ph=self.mlp_input_ph,
                           reuse=False,
                           mlp_config=mlp_config,
                           input_norm=input_norm,
                           output_norm=output_norm,
                           output_high=output_high - output_low,
                           output_low=output_low - output_high,
                           name_scope=name_scope,
                           net_name='mlp')
        assert self.mlp_net.output.shape[1] == env_spec.flat_obs_dim

        self.parameters = TensorflowParameters(tf_var_list=self.mlp_net.var_list,
                                               name='mlp_continuous_dynamics_model',
                                               rest_parameters=dict(l1_norm_scale=l1_norm_scale,
                                                                    l2_norm_scale=l2_norm_scale,
                                                                    output_low=output_low,
                                                                    output_high=output_high,
                                                                    input_norm=input_norm,
                                                                    learning_rate=learning_rate),
                                               auto_init=False)
        with tf.variable_scope(self.name_scope):
            with tf.variable_scope('train'):
                self.new_state_output = self.mlp_net.output + self.state_input
                self.loss, self.optimizer, self.optimize_op = self._setup_loss(l1_norm_scale=l1_norm_scale,
                                                                               l2_norm_scale=l2_norm_scale)
        train_var_list = get_tf_collection_var_list(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='{}/train'.format(
                                                        self.name_scope)) + self.optimizer.variables()

        self.parameters.set_tf_var_list(sorted(list(set(train_var_list)), key=lambda x: x.name))
        DerivableDynamics.__init__(self,
                                   input_node_dict=dict(state_input=self.state_input,
                                                        action_action_input=self.action_input),
                                   output_node_dict=dict(new_state_output=self.new_state_output))
        PlaceholderInput.__init__(self,
                                  inputs=(self.state_input, self.action_input, self.delta_state_label_ph),
                                  parameters=self.parameters)

    def init(self, source_obj=None):
        self.parameters.init()
        if source_obj:
            self.copy_from(obj=source_obj)

    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        if 'sess' in kwargs:
            tf_sess = kwargs['sess']
        else:
            tf_sess = tf.get_default_session()

        if len(state.shape) < 2:
            state = np.expand_dims(state, 0)
        if len(action.shape) < 2:
            action = np.expand_dims(action, 0)
        new_state = tf_sess.run(self.new_state_output,
                                feed_dict={
                                    self.action_input: action,
                                    self.state_input: state
                                })
        return np.clip(np.squeeze(new_state), self.parameters('output_low'), self.parameters('output_high'))

    def _setup_loss(self, l1_norm_scale, l2_norm_scale):
        l1_l2 = tf_contrib.layers.l1_l2_regularizer(scale_l1=l1_norm_scale,
                                                    scale_l2=l2_norm_scale)
        loss = tf.reduce_sum((self.mlp_net.output - self.delta_state_label_ph) ** 2) + \
               tf_contrib.layers.apply_regularization(l1_l2, weights_list=self.parameters('tf_var_list'))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters('learning_rate'))
        optimize_op = optimizer.minimize(loss=loss, var_list=self.parameters('tf_var_list'))
        return loss, optimizer, optimize_op

    @typechecked
    def train(self, batch_data: TransitionData, **kwargs) -> dict:

        tf_sess = kwargs['sess'] if ('sess' in kwargs and kwargs['sess']) else tf.get_default_session()
        train_iter = self.parameters('train_iter') if 'train_iter' not in kwargs else kwargs['train_iter']
        feed_dict = {
            self.state_input: batch_data.state_set,
            self.action_input: flatten_n(self.env_space.action_space, batch_data.action_set),
            self.delta_state_label_ph: batch_data.new_state_set - batch_data.state_set,
            **self.parameters.return_tf_parameter_feed_dict()
        }
        average_loss = 0.0

        for i in range(train_iter):
            loss, _ = tf_sess.run([self.loss, self.optimize_op],
                                  feed_dict=feed_dict)
            average_loss += loss
        return dict(average_loss=average_loss / train_iter)

    def save(self, *args, **kwargs):
        return PlaceholderInput.save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        return PlaceholderInput.load(self, *args, **kwargs)
