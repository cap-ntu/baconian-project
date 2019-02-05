from src.envs.env_spec import EnvSpec
from src.rl.algo.model_based.models.dynamics_model import DynamicsModel
import tensorflow as tf
from src.tf.tf_parameters import TensorflowParameters
from src.tf.mlp import MLP
import tensorflow.contrib as tf_contrib
from src.common.sampler.sample_data import TransitionData
from typeguard import typechecked
from src.misc.special import *
from src.envs.util import *


class ContinuousMLPDynamicsModel(DynamicsModel):
    def __init__(self, env_spec: EnvSpec,
                 name_scope: str,
                 mlp_config: list,
                 input_norm: bool,
                 learning_rate: float,
                 l1_norm_scale: float,
                 l2_norm_scale: float,
                 output_norm: bool,
                 output_low: (list, np.ndarray, None),
                 output_high: (list, np.ndarray, None),
                 init_state=None):
        super().__init__(env_spec, init_state)
        self.mlp_config = mlp_config
        self.name_scope = name_scope
        with tf.variable_scope(self.name_scope):
            self.state_ph = tf.placeholder(shape=[None, env_spec.flat_obs_dim], dtype=tf.float32, name='state_ph')
            self.action_ph = tf.placeholder(shape=[None, env_spec.flat_action_dim], dtype=tf.float32, name='action_ph')
            self.mlp_input_ph = tf.concat([self.state_ph, self.action_ph], axis=1, name='state_action_input')
            self.delta_state_label_ph = tf.placeholder(shape=[None, env_spec.flat_obs_dim], dtype=tf.float32,
                                                       name='delta_state_label_ph')
        self.mlp_net = MLP(input_ph=self.mlp_input_ph,
                           mlp_config=mlp_config,
                           input_norm=input_norm,
                           output_norm=output_norm,
                           output_high=output_high - output_low,
                           output_low=output_low - output_high,
                           name_scope=name_scope,
                           net_name='mlp')
        assert self.mlp_net.output.shape[1] == env_spec.flat_obs_dim
        parameters = TensorflowParameters(tf_var_list=self.mlp_net.var_list,
                                          name='mlp_continuous_dynamics_model',
                                          rest_parameters=dict(l1_norm_scale=l1_norm_scale,
                                                               l2_norm_scale=l2_norm_scale,
                                                               output_low=output_low,
                                                               output_high=output_high,
                                                               input_norm=input_norm,
                                                               learning_rate=learning_rate),
                                          auto_init=False)

        self.parameters = parameters
        with tf.variable_scope(self.name_scope):
            with tf.variable_scope('train'):
                self.new_state_output = self.mlp_net.output + self.state_ph
                self.loss, self._optimizer, self.optimize_op = self._setup_loss(l1_norm_scale=l1_norm_scale,
                                                                                l2_norm_scale=l2_norm_scale)
        train_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}/train'.format(self.name_scope))
        self.parameters.set_tf_var_list(train_var_list + self.parameters('tf_var_list'))
        # todo super __init__ may override the self.parameters
        super(ContinuousMLPDynamicsModel, self).__init__(env_spec=env_spec, parameters=parameters)

    def init(self, source_obj=None):
        self.parameters.init()
        if source_obj:
            self.copy(obj=source_obj)

    def copy(self, obj: DynamicsModel) -> bool:
        super().copy(obj)
        self.parameters.copy_from(source_parameter=obj.parameters)
        return True

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
                                    self.action_ph: action,
                                    self.state_ph: state
                                })
        return np.clip(np.squeeze(new_state), self.parameters('output_low'), self.parameters('output_high'))

    def _setup_loss(self, l1_norm_scale, l2_norm_scale):
        l1_l2 = tf_contrib.layers.l1_l2_regularizer(scale_l1=l1_norm_scale,
                                                    scale_l2=l2_norm_scale)
        loss = tf.reduce_sum((self.mlp_net.output - self.delta_state_label_ph) ** 2) + \
               tf_contrib.layers.apply_regularization(l1_l2, weights_list=self.parameters('tf_var_list'))
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.parameters('learning_rate'))
        optimize_op = optimizer.minimize(loss=loss, var_list=self.parameters('tf_var_list'))
        return loss, optimizer, optimize_op

    @typechecked
    def train(self, batch_data: TransitionData, **kwargs) -> dict:

        tf_sess = kwargs['sess'] if ('sess' in kwargs and kwargs['sess']) else tf.get_default_session()
        train_iter = self.parameters('train_iter') if 'train_iter' not in kwargs else kwargs['train_iter']
        feed_dict = {
            self.state_ph: batch_data.state_set,
            self.action_ph: flatten_n(self.env_space.action_space, batch_data.action_set),
            self.delta_state_label_ph: batch_data.new_state_set - batch_data.state_set,
            **self.parameters.return_tf_parameter_feed_dict()
        }
        average_loss = 0.0

        for i in range(train_iter):
            loss, _ = tf_sess.run([self.loss, self.optimize_op],
                                  feed_dict=feed_dict)
            average_loss += loss

        return dict(average_loss=average_loss / train_iter)