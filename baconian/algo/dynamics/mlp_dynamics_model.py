from baconian.common.special import flatten_n
from baconian.core.core import EnvSpec
from baconian.algo.dynamics.dynamics_model import GlobalDynamicsModel, DifferentiableDynamics, TrainableDyanmicsModel
import tensorflow as tf
from baconian.tf.tf_parameters import ParametersWithTensorflowVariable
from baconian.tf.mlp import MLP
from baconian.common.sampler.sample_data import TransitionData
from typeguard import typechecked
import numpy as np
from baconian.tf.util import *
from baconian.algo.misc.placeholder_input import PlaceholderInput
import overrides
from baconian.common.error import *
from baconian.common.logging import record_return_decorator
from baconian.core.status import register_counter_info_to_status_decorator, StatusWithSubInfo
from baconian.common.spaces.box import Box
from baconian.common.data_pre_processing import DataScaler, IdenticalDataScaler


class ContinuousMLPGlobalDynamicsModel(GlobalDynamicsModel, DifferentiableDynamics, PlaceholderInput,
                                       TrainableDyanmicsModel):
    STATUS_LIST = GlobalDynamicsModel.STATUS_LIST + ('TRAIN',)
    INIT_STATUS = 'CREATED'

    def __init__(self, env_spec: EnvSpec,
                 name_scope: str,
                 name: str,
                 mlp_config: list,
                 learning_rate: float,
                 state_input_scaler: DataScaler = None,
                 action_input_scaler: DataScaler = None,
                 output_delta_state_scaler: DataScaler = None,
                 init_state=None):
        if not isinstance(env_spec.obs_space, Box):
            raise TypeError('ContinuousMLPGlobalDynamicsModel only support to predict state that hold space Box type')
        GlobalDynamicsModel.__init__(self,
                                     env_spec=env_spec,
                                     parameters=None,
                                     name=name,
                                     state_input_scaler=state_input_scaler,
                                     action_input_scaler=action_input_scaler,
                                     init_state=init_state)
        with tf.variable_scope(name_scope):
            state_input = tf.placeholder(shape=[None, env_spec.flat_obs_dim], dtype=tf.float32, name='state_ph')
            action_input = tf.placeholder(shape=[None, env_spec.flat_action_dim], dtype=tf.float32,
                                          name='action_ph')
            mlp_input_ph = tf.concat([state_input, action_input], axis=1, name='state_action_input')
            delta_state_label_ph = tf.placeholder(shape=[None, env_spec.flat_obs_dim], dtype=tf.float32,
                                                  name='delta_state_label_ph')
        mlp_net = MLP(input_ph=mlp_input_ph,
                      reuse=False,
                      mlp_config=mlp_config,
                      name_scope=name_scope,
                      net_name='mlp')
        if mlp_net.output.shape[1] != env_spec.flat_obs_dim:
            raise InappropriateParameterSetting(
                "mlp output dims {} != env spec obs dim {}".format(mlp_net.output.shape[1], env_spec.flat_obs_dim))

        parameters = ParametersWithTensorflowVariable(tf_var_list=mlp_net.var_list,
                                                      name=name + '_''mlp_continuous_dynamics_model',
                                                      rest_parameters=dict(learning_rate=learning_rate))

        DifferentiableDynamics.__init__(self,
                                        input_node_dict=dict(state_input=state_input,
                                                             action_action_input=action_input),
                                        output_node_dict=dict(delta_state_output=mlp_net.output))
        PlaceholderInput.__init__(self, parameters=parameters)

        self.mlp_config = mlp_config
        self.name_scope = name_scope
        self.action_input = action_input
        self.state_input = state_input
        self.mlp_input_ph = mlp_input_ph
        self.delta_state_label_ph = delta_state_label_ph
        self.delta_state_output = mlp_net.output
        self.mlp_net = mlp_net
        self.output_delta_state_scaler = output_delta_state_scaler if output_delta_state_scaler else IdenticalDataScaler(
            dims=self.env_spec.flat_obs_dim)
        self._status = StatusWithSubInfo(obj=self)

        with tf.variable_scope(name_scope):
            with tf.variable_scope('train'):
                self.loss, self.optimizer, self.optimize_op = self._setup_loss()
        train_var_list = get_tf_collection_var_list(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='{}/train'.format(
                                                        name_scope)) + self.optimizer.variables()

        self.parameters.set_tf_var_list(sorted(list(set(train_var_list)), key=lambda x: x.name))

    def init(self, source_obj=None):
        self.parameters.init()
        if source_obj:
            self.copy_from(obj=source_obj)
        GlobalDynamicsModel.init(self)

    @register_counter_info_to_status_decorator(increment=1, info_key='step')
    def step(self, action: np.ndarray, state=None, **kwargs_for_transit):
        return super().step(action, state, **kwargs_for_transit)

    @record_return_decorator(which_recorder='self')
    @register_counter_info_to_status_decorator(increment=1, info_key='train_counter', under_status='TRAIN')
    def train(self, batch_data: TransitionData, **kwargs) -> dict:
        self.set_status('TRAIN')

        self.state_input_scaler.update_scaler(batch_data.state_set)
        self.action_input_scaler.update_scaler(batch_data.action_set)
        self.output_delta_state_scaler.update_scaler(batch_data.new_state_set - batch_data.state_set)

        tf_sess = kwargs['sess'] if ('sess' in kwargs and kwargs['sess']) else tf.get_default_session()
        train_iter = self.parameters('train_iter') if 'train_iter' not in kwargs else kwargs['train_iter']
        feed_dict = {
            self.state_input: self.state_input_scaler.process(batch_data.state_set),
            self.action_input: self.action_input_scaler.process(
                flatten_n(self.env_spec.action_space, batch_data.action_set)),
            self.delta_state_label_ph: self.output_delta_state_scaler.process(
                batch_data.new_state_set - batch_data.state_set),
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

    @overrides.overrides
    def copy_from(self, obj: PlaceholderInput) -> bool:
        return PlaceholderInput.copy_from(self, obj)

    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        state = self.state_input_scaler.process(
            np.array(state).reshape(self.env_spec.obs_shape)) if state is not None else self.state
        action = self.action_input_scaler.process(self.env_spec.flat_action(action))

        if 'sess' in kwargs:
            tf_sess = kwargs['sess']
        else:
            tf_sess = tf.get_default_session()

        if len(state.shape) < 2:
            state = np.expand_dims(state, 0)
        if len(action.shape) < 2:
            action = np.expand_dims(action, 0)
        delta_state = tf_sess.run(self.delta_state_output,
                                  feed_dict={
                                      self.action_input: action,
                                      self.state_input: state
                                  })
        new_state = np.clip(
            np.squeeze(self.output_delta_state_scaler.inverse_process(data=np.squeeze(delta_state)) + state),
            self.env_spec.obs_space.low, self.env_spec.obs_space.high)
        return new_state

    def _setup_loss(self):
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.name_scope)
        loss = tf.reduce_sum((self.mlp_net.output - self.delta_state_label_ph) ** 2)
        if len(reg_loss) > 0:
            loss += tf.reduce_sum(reg_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters('learning_rate'))
        optimize_op = optimizer.minimize(loss=loss, var_list=self.parameters('tf_var_list'))
        return loss, optimizer, optimize_op
