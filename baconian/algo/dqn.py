from baconian.common.special import flatten_n
from baconian.algo.rl_algo import ModelFreeAlgo, OffPolicyAlgo
from baconian.config.dict_config import DictConfig
from typeguard import typechecked
from baconian.core.util import init_func_arg_record_decorator
from baconian.tf.util import *
from baconian.algo.misc.replay_buffer import UniformRandomReplayBuffer, BaseReplayBuffer, PrioritisedReplayBuffer
import tensorflow as tf
import numpy as np
from baconian.common.sampler.sample_data import TransitionData
from baconian.tf.tf_parameters import ParametersWithTensorflowVariable
from baconian.config.global_config import GlobalConfig
from baconian.common.misc import *
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.common.logging import record_return_decorator
from baconian.core.status import register_counter_info_to_status_decorator
from baconian.algo.misc.placeholder_input import MultiPlaceholderInput
from baconian.common.error import *


class DQN(ModelFreeAlgo, OffPolicyAlgo, MultiPlaceholderInput):
    required_key_dict = DictConfig.load_json(file_path=GlobalConfig().DEFAULT_DQN_REQUIRED_KEY_LIST)

    @init_func_arg_record_decorator()
    @typechecked
    def __init__(self,
                 env_spec,
                 config_or_config_dict: (DictConfig, dict),
                 value_func: MLPQValueFunction,
                 schedule_param_list=None,
                 name: str = 'dqn',
                 replay_buffer=None):
        ModelFreeAlgo.__init__(self, env_spec=env_spec, name=name)
        self.config = construct_dict_config(config_or_config_dict, self)

        if replay_buffer:
            assert issubclass(replay_buffer, BaseReplayBuffer)
            self.replay_buffer = replay_buffer
        else:
            self.replay_buffer = UniformRandomReplayBuffer(limit=self.config('REPLAY_BUFFER_SIZE'),
                                                           action_shape=self.env_spec.action_shape,
                                                           observation_shape=self.env_spec.obs_shape)
        self.q_value_func = value_func
        self.state_input = self.q_value_func.state_input
        self.action_input = self.q_value_func.action_input
        self.update_target_q_every_train = self.config('UPDATE_TARGET_Q_FREQUENCY') if 'UPDATE_TARGET_Q_FREQUENCY' in \
                                                                                       self.config.config_dict else 1
        self.parameters = ParametersWithTensorflowVariable(tf_var_list=[],
                                                           rest_parameters=dict(),
                                                           to_scheduler_param_tuple=schedule_param_list,
                                                           name='{}_param'.format(name),
                                                           source_config=self.config,
                                                           require_snapshot=False)

        with tf.variable_scope(name):
            self.reward_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.next_state_input = tf.placeholder(shape=[None, self.env_spec.flat_obs_dim], dtype=tf.float32)
            self.done_input = tf.placeholder(shape=[None, 1], dtype=tf.bool)
            self.target_q_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            done = tf.cast(self.done_input, dtype=tf.float32)
            self.target_q_value_func = self.q_value_func.make_copy(name_scope='{}_targe_q_value_net'.format(name),
                                                                   name='{}_targe_q_value_net'.format(name),
                                                                   reuse=False)
            self.predict_q_value = (1. - done) * self.config('GAMMA') * self.target_q_input + self.reward_input
            self.td_error = self.predict_q_value - self.q_value_func.q_tensor
            with tf.variable_scope('train'):
                self.q_value_func_loss, self.optimizer, self.update_q_value_func_op = self._set_up_loss()
                self.update_target_q_value_func_op = self._set_up_target_update()
        var_list = get_tf_collection_var_list(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope='{}/train'.format(name)) + self.optimizer.variables()
        self.parameters.set_tf_var_list(tf_var_list=sorted(list(set(var_list)), key=lambda x: x.name))

        MultiPlaceholderInput.__init__(self,
                                       sub_placeholder_input_list=[dict(obj=self.q_value_func,
                                                                        attr_name='q_value_func',
                                                                        ),
                                                                   dict(obj=self.target_q_value_func,
                                                                        attr_name='target_q_value_func')],
                                       parameters=self.parameters)

    @register_counter_info_to_status_decorator(increment=1, info_key='init', under_status='INITED')
    def init(self, sess=None, source_obj=None):
        super().init()
        self.q_value_func.init()
        self.target_q_value_func.init(source_obj=self.q_value_func)
        self.parameters.init()
        if source_obj:
            self.copy_from(source_obj)

    @record_return_decorator(which_recorder='self')
    @register_counter_info_to_status_decorator(increment=1, info_key='train_counter', under_status='TRAIN')
    def train(self, batch_data=None, train_iter=None, sess=None, update_target=True) -> dict:
        super(DQN, self).train()
        self.recorder.record()
        if batch_data and not isinstance(batch_data, TransitionData):
            raise TypeError()

        tf_sess = sess if sess else tf.get_default_session()
        train_iter = self.parameters("TRAIN_ITERATION") if not train_iter else train_iter
        average_loss = 0.0

        for i in range(train_iter):
            train_data = self.replay_buffer.sample(
                batch_size=self.parameters('BATCH_SIZE')) if batch_data is None else batch_data

            _, target_q_val_on_new_s = self.predict_target_with_q_val(obs=train_data.new_state_set,
                                                                      batch_flag=True)
            target_q_val_on_new_s = np.expand_dims(target_q_val_on_new_s, axis=1)
            assert target_q_val_on_new_s.shape[0] == train_data.state_set.shape[0]
            feed_dict = {
                self.reward_input: np.reshape(train_data.reward_set, [-1, 1]),
                self.action_input: flatten_n(self.env_spec.action_space, train_data.action_set),
                self.state_input: train_data.state_set,
                self.done_input: np.reshape(train_data.done_set, [-1, 1]),
                self.target_q_input: target_q_val_on_new_s,
                **self.parameters.return_tf_parameter_feed_dict()
            }
            res, _ = tf_sess.run([self.q_value_func_loss, self.update_q_value_func_op],
                                 feed_dict=feed_dict)
            average_loss += res

        average_loss /= train_iter
        if update_target is True and self.get_status()['train_counter'] % self.update_target_q_every_train == 0:
            tf_sess.run(self.update_target_q_value_func_op,
                        feed_dict=self.parameters.return_tf_parameter_feed_dict())
        return dict(average_loss=average_loss)

    @register_counter_info_to_status_decorator(increment=1, info_key='test_counter', under_status='TEST')
    def test(self, *arg, **kwargs):
        return super().test(*arg, **kwargs)

    @register_counter_info_to_status_decorator(increment=1, info_key='predict_counter')
    def predict(self, obs: np.ndarray, sess=None, batch_flag: bool = False):
        if batch_flag:
            action, q_val = self._predict_batch_action(obs=obs,
                                                       q_value_tensor=self.q_value_func.q_tensor,
                                                       action_ph=self.action_input,
                                                       state_ph=self.state_input,
                                                       sess=sess)
        else:
            action, q_val = self._predict_action(obs=obs,
                                                 q_value_tensor=self.q_value_func.q_tensor,
                                                 action_ph=self.action_input,
                                                 state_ph=self.state_input,
                                                 sess=sess)
        if not batch_flag:
            return int(action)
        else:
            return action.astype(np.int).tolist()

    def predict_target_with_q_val(self, obs: np.ndarray, sess=None, batch_flag: bool = False):
        if batch_flag:
            action, q_val = self._predict_batch_action(obs=obs,
                                                       q_value_tensor=self.target_q_value_func.q_tensor,
                                                       action_ph=self.target_q_value_func.action_input,
                                                       state_ph=self.target_q_value_func.state_input,
                                                       sess=sess)
        else:
            action, q_val = self._predict_action(obs=obs,
                                                 q_value_tensor=self.target_q_value_func.q_tensor,
                                                 action_ph=self.target_q_value_func.action_input,
                                                 state_ph=self.target_q_value_func.state_input,
                                                 sess=sess)
        return action, q_val

    @register_counter_info_to_status_decorator(increment=1, info_key='append_to_memory')
    def append_to_memory(self, samples: TransitionData):
        iter_samples = samples.return_generator()
        data_count = 0
        for obs0, obs1, action, reward, terminal1 in iter_samples:
            self.replay_buffer.append(obs0=obs0,
                                      obs1=obs1,
                                      action=action,
                                      reward=reward,
                                      terminal1=terminal1)
            data_count += 1
        self._status.update_info(info_key='replay_buffer_data_total_count', increment=data_count)

    @record_return_decorator(which_recorder='self')
    def save(self, global_step, save_path=None, name=None, **kwargs):
        save_path = save_path if save_path else GlobalConfig().DEFAULT_MODEL_CHECKPOINT_PATH
        name = name if name else self.name
        MultiPlaceholderInput.save(self, save_path=save_path, global_step=global_step, name=name, **kwargs)
        return dict(check_point_save_path=save_path, check_point_save_global_step=global_step,
                    check_point_save_name=name)

    @record_return_decorator(which_recorder='self')
    def load(self, path_to_model, model_name, global_step=None, **kwargs):
        MultiPlaceholderInput.load(self, path_to_model, model_name, global_step, **kwargs)
        return dict(check_point_load_path=path_to_model, check_point_load_global_step=global_step,
                    check_point_load_name=model_name)

    def _predict_action(self, obs: np.ndarray, q_value_tensor: tf.Tensor, action_ph: tf.Tensor, state_ph: tf.Tensor,
                        sess=None):
        if self.env_spec.obs_space.contains(obs) is False:
            raise StateOrActionOutOfBoundError("obs {} out of bound {}".format(obs, self.env_spec.obs_space.bound()))
        obs = repeat_ndarray(obs, repeats=self.env_spec.flat_action_dim)
        tf_sess = sess if sess else tf.get_default_session()
        feed_dict = {action_ph: generate_n_actions_hot_code(n=self.env_spec.flat_action_dim),
                     state_ph: obs, **self.parameters.return_tf_parameter_feed_dict()}
        res = tf_sess.run([q_value_tensor],
                          feed_dict=feed_dict)[0]
        return np.argmax(res, axis=0), np.max(res, axis=0)

    def _predict_batch_action(self, obs: np.ndarray, q_value_tensor: tf.Tensor, action_ph: tf.Tensor,
                              state_ph: tf.Tensor, sess=None):
        actions = []
        q_values = []
        for obs_i in obs:
            action, q_val = self._predict_action(obs=obs_i,
                                                 q_value_tensor=q_value_tensor,
                                                 action_ph=action_ph,
                                                 state_ph=state_ph,
                                                 sess=sess)
            actions.append(np.argmax(action, axis=0))
            q_values.append(np.max(q_val, axis=0))
        return np.array(actions), np.array(q_values)

    def _set_up_loss(self):
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.q_value_func.name_scope)
        loss = tf.reduce_sum((self.predict_q_value - self.q_value_func.q_tensor) ** 2)
        if len(reg_loss) > 0:
            loss += tf.reduce_sum(reg_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters('LEARNING_RATE'))
        optimize_op = optimizer.minimize(loss=loss, var_list=self.q_value_func.parameters('tf_var_list'))
        return loss, optimizer, optimize_op

    def _set_up_target_update(self):
        op = []
        for var, target_var in zip(self.q_value_func.parameters('tf_var_list'),
                                   self.target_q_value_func.parameters('tf_var_list')):
            ref_val = self.parameters('DECAY') * target_var + (1.0 - self.parameters('DECAY')) * var
            op.append(tf.assign(target_var, ref_val))
        return op

    def _evaluate_td_error(self, sess=None):
        # tf_sess = sess if sess else tf.get_default_session()
        # feed_dict = {
        #     self.reward_input: train_data.reward_set,
        #     self.action_input: flatten_n(self.env_spec.action_space, train_data.action_set),
        #     self.state_input: train_data.state_set,
        #     self.done_input: train_data.done_set,
        #     self.target_q_input: target_q_val_on_new_s,
        #     **self.parameters.return_tf_parameter_feed_dict()
        # }
        # td_loss = tf_sess.run([self.td_error], feed_dict=feed_dict)
        pass