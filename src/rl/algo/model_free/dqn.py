# Date: 11/16/18
# Author: Luke
# Project: ModelBasedRLFramework
import os
from src.rl.algo.algo import ModelFreeAlgo
from src.core.config import Config
from typeguard import typechecked
from src.rl.value_func import *
from src.envs.env import Env
from src.rl.algo.util.replay_buffer import UniformRandomReplayBuffer, BaseReplayBuffer
from src.util.required_keys import SRC_UTIL_REQUIRED_KEYS
import tensorflow as tf
import tensorflow.contrib as tfcontrib
import numpy as np
from src.misc import *
from src.core.global_config import GlobalConfig
from src.common.sampler.sample_data import TransitionData


# todo
# 1. how to define the action iterator, need one hot ?
# 2. how to define the action dim and flat dim


class DQN(ModelFreeAlgo):
    required_key_list = Config.load_json(file_path=os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                                'dqn.json'))

    @typechecked
    def __init__(self,
                 env: GlobalConfig.DEFAULT_ALLOWED_GYM_ENV_TYPE + (Env,),
                 config_or_config_dict: (Config, dict),
                 # todo check the type list
                 # todo bug on mlp value function and its placeholder which is crushed with the dqn placeholder
                 value_func: MLPQValueFunction,
                 replay_buffer=None):
        # todo add the action iterator
        super().__init__(env=env)
        if isinstance(config_or_config_dict, dict):
            config = Config(required_key_dict=self.required_key_list, config_dict=config_or_config_dict)
        elif isinstance(config_or_config_dict, Config):
            config = config_or_config_dict
        else:
            raise TypeError('Type {} is not supported, use dict or Config'.format(type(config_or_config_dict).__name__))

        self.config = config
        if replay_buffer:
            assert issubclass(replay_buffer, BaseReplayBuffer)
            self.replay_buffer = replay_buffer
        else:
            self.replay_buffer = UniformRandomReplayBuffer(limit=self.config('REPLAY_BUFFER_SIZE'),
                                                           action_shape=(self.env_spec.flat_action_dim,),
                                                           observation_shape=(self.env_spec.flat_obs_dim,))
        self.q_value_func = value_func
        self.state_input = self.q_value_func.state_ph
        self.action_input = self.q_value_func.action_ph
        with tf.variable_scope('dqn'):
            self.reward_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.next_state_input = tf.placeholder(shape=[None, self.env_spec.flat_obs_dim], dtype=tf.float32)
            self.done_input = tf.placeholder(shape=[None, 1], dtype=tf.bool)
            self.target_q_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            done = tf.cast(self.done_input, dtype=tf.float32)
            self.target_q_value_func = self.q_value_func.make_copy(name_scope='dqn_targe_q_value_net')
            self.predict_q_value = (1. - done) * self.config('GAMMA') * self.target_q_input + self.reward_input
            with tf.variable_scope('train'):
                self.q_value_func_loss, _, self.update_q_value_func_op = self._set_up_loss()
                self.update_target_q_value_func_op = self._set_up_target_update()
        # todo handle the var list problem
        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dqn/train')

    def init(self, sess=None):
        self.q_value_func.init()
        self.target_q_value_func.init(source_obj=self.q_value_func)
        tf_sess = sess if sess else tf.get_default_session()
        tf_sess.run(tf.variables_initializer(var_list=self.var_list))

    @typechecked
    def train(self, batch_data: (dict, None), train_iter=10, sess=None, update_target=False):
        if not batch_data:
            batch_data = self.replay_buffer.sample(batch_size=self.config('BATCH_SIZE'))
        average_loss = 0.0
        tf_sess = sess if sess else tf.get_default_session()
        _, target_q_val_on_new_s = self.predict_target_with_q_val(obs=batch_data['obs0'],
                                                                  batch_flag=True)
        target_q_val_on_new_s = np.expand_dims(target_q_val_on_new_s, axis=1)
        assert target_q_val_on_new_s.shape[0] == batch_data['obs0'].shape[0]
        for i in range(train_iter):
            res, _ = tf_sess.run([self.q_value_func_loss, self.update_q_value_func_op],
                                 feed_dict={
                                     self.reward_input: batch_data['rewards'],
                                     self.action_input: batch_data['actions'],
                                     self.state_input: batch_data['obs0'],
                                     self.done_input: batch_data['terminals1'],
                                     self.target_q_input: target_q_val_on_new_s
                                 })
            average_loss += res
        average_loss /= train_iter
        if update_target is True:
            tf_sess.run(self.update_target_q_value_func_op)
        return average_loss

    def evaluate(self, *arg, **kwargs):
        raise NotImplementedError

    @typechecked
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

    @typechecked
    def predict_target_with_q_val(self, obs: np.ndarray, sess=None, batch_flag: bool = False):
        if batch_flag:
            action, q_val = self._predict_batch_action(obs=obs,
                                                       q_value_tensor=self.target_q_value_func.q_tensor,
                                                       action_ph=self.target_q_value_func.action_ph,
                                                       state_ph=self.target_q_value_func.state_ph,
                                                       sess=sess)
        else:
            action, q_val = self._predict_action(obs=obs,
                                                 q_value_tensor=self.q_value_func.q_tensor,
                                                 action_ph=self.target_q_value_func.action_ph,
                                                 state_ph=self.target_q_value_func.state_ph,
                                                 sess=sess)
        return action, q_val

    def append_to_memory(self, samples: TransitionData):
        for obs0, obs1, action, reward, terminal1 in zip(samples.state_set, samples.new_state_set, samples.action_set,
                                                         samples.reward_set, samples.done_set):
            self.replay_buffer.append(obs0=obs0,
                                      obs1=obs1,
                                      action=action,
                                      reward=reward,
                                      terminal1=terminal1)

    def _predict_action(self, obs: np.ndarray, q_value_tensor: tf.Tensor, action_ph: tf.Tensor, state_ph: tf.Tensor,
                        sess=None):
        assert self.env_spec.obs_space.contains(obs)
        obs = repeat_ndarray(obs, repeats=self.env_spec.flat_action_dim)
        tf_sess = sess if sess else tf.get_default_session()
        res = tf_sess.run([q_value_tensor],
                          feed_dict={action_ph: generate_n_actions_hot_code(n=self.env_spec.flat_action_dim),
                                     state_ph: obs})[0]
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
        l1_l2 = tfcontrib.layers.l1_l2_regularizer(scale_l1=self.config('Q_NET_L1_NORM_SCALE'),
                                                   scale_l2=self.config('Q_NET_L2_NORM_SCALE'))
        loss = tf.reduce_sum((self.predict_q_value - self.q_value_func.q_tensor) ** 2) + \
               tfcontrib.layers.apply_regularization(l1_l2, weights_list=self.q_value_func.parameters())
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.config('LEARNING_RATE'))
        optimize_op = optimizer.minimize(loss=loss, var_list=self.q_value_func.parameters())
        return loss, optimizer, optimize_op

    def _set_up_target_update(self):
        op = []
        for var, target_var in zip(self.q_value_func.parameters(), self.target_q_value_func.parameters()):
            ref_val = self.config('DECAY') * target_var + (1.0 - self.config('DECAY')) * var
            op.append(tf.assign(target_var, ref_val))
        return op
