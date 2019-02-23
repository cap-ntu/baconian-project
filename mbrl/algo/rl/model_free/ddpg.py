from mbrl.envs.env_spec import EnvSpec
from mbrl.algo.rl.rl_algo import ModelFreeAlgo, OffPolicyAlgo
from mbrl.config.dict_config import DictConfig
from typeguard import typechecked
from mbrl.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from mbrl.algo.rl.util.replay_buffer import UniformRandomReplayBuffer, BaseReplayBuffer
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
from mbrl.common.sampler.sample_data import TransitionData
from mbrl.tf.tf_parameters import TensorflowParameters
from mbrl.config.global_config import GlobalConfig
from mbrl.common.special import *
from mbrl.algo.rl.policy.deterministic_mlp import DeterministicMLPPolicy
from mbrl.tf.util import *
from mbrl.common.misc import *
from mbrl.common.util.recorder import record_return_decorator
from mbrl.core.status import register_counter_status_decorator


class DDPG(ModelFreeAlgo, OffPolicyAlgo):
    required_key_list = DictConfig.load_json(file_path=GlobalConfig.DEFAULT_DDPG_REQUIRED_KEY_LIST)

    @typechecked
    def __init__(self,
                 env_spec: EnvSpec,
                 config_or_config_dict: (DictConfig, dict),
                 # todo bug on mlp value function and its placeholder which is crushed with the dqn placeholder
                 value_func: MLPQValueFunction,
                 policy: DeterministicMLPPolicy,
                 adaptive_learning_rate=False,
                 name='ddpg',
                 replay_buffer=None):
        super(DDPG, self).__init__(env_spec, name=name)
        config = construct_dict_config(config_or_config_dict, self)

        self.config = config
        self.actor = policy
        self.target_actor = self.actor.make_copy(name_scope='target_actor', reuse=False)
        self.critic = value_func
        self.target_critic = self.critic.make_copy(name_scope='target_critic', reuse=False)

        self.state_input = self.actor.state_input

        if replay_buffer:
            assert issubclass(replay_buffer, BaseReplayBuffer)
            self.replay_buffer = replay_buffer
        else:
            self.replay_buffer = UniformRandomReplayBuffer(limit=self.config('REPLAY_BUFFER_SIZE'),
                                                           action_shape=self.env_spec.action_shape,
                                                           observation_shape=self.env_spec.obs_shape)

        self.adaptive_learning_rate = adaptive_learning_rate
        to_ph_parameter_dict = dict()
        with tf.variable_scope(name):
            if adaptive_learning_rate is not False:
                to_ph_parameter_dict['ACTOR_LEARNING_RATE'] = tf.placeholder(shape=(), dtype=tf.float32)
                to_ph_parameter_dict['CRITIC_LEARNING_RATE'] = tf.placeholder(shape=(), dtype=tf.float32)

        self.parameters = TensorflowParameters(tf_var_list=[],
                                               rest_parameters=dict(),
                                               to_ph_parameter_dict=to_ph_parameter_dict,
                                               name='ddpg_param',
                                               auto_init=False,
                                               source_config=config,
                                               require_snapshot=False)
        # todo check this reuse utility
        self._critic_with_actor_output = self.critic.make_copy(reuse=True,
                                                               action_input=self.actor.action_tensor)
        self._target_critic_with_target_actor_output = self.target_critic.make_copy(reuse=True,
                                                                                    action_input=self.target_actor.action_tensor)

        with tf.variable_scope(name):
            self.reward_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.next_state_input = tf.placeholder(shape=[None, self.env_spec.flat_obs_dim], dtype=tf.float32)
            self.done_input = tf.placeholder(shape=[None, 1], dtype=tf.bool)
            self.target_q_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            done = tf.cast(self.done_input, dtype=tf.float32)
            self.predict_q_value = (1. - done) * self.config('GAMMA') * self.target_q_input + self.reward_input
            with tf.variable_scope('train'):
                self.critic_loss, self.critic_update_op, self.target_critic_update_op, self.critic_optimizer = self._setup_critic_loss()
                self.actor_loss, self.actor_update_op, self.target_actor_update_op, self.action_optimizer = self._set_up_actor_loss()

        var_list = get_tf_collection_var_list(
            '{}/train'.format(name)) + self.critic_optimizer.variables() + self.action_optimizer.variables()
        self.parameters.set_tf_var_list(tf_var_list=sorted(list(set(var_list)), key=lambda x: x.name))

    @register_counter_status_decorator(increment=1, key='init')
    def init(self, sess=None):
        self.actor.init()
        self.critic.init()
        self.target_actor.init()
        self.target_critic.init(source_obj=self.critic)
        # tf_sess = sess if sess else tf.get_default_session()
        # feed_dict = self.parameters.return_tf_parameter_feed_dict()
        # tf_sess.run(tf.variables_initializer(var_list=self.parameters('tf_var_list')),
        #             feed_dict=feed_dict)
        self.parameters.init()
        super().init()

    @record_return_decorator(which_recorder='self')
    @register_counter_status_decorator(increment=1, key='train')
    @typechecked
    def train(self, batch_data=None, train_iter=None, sess=None, update_target=True) -> dict:
        super(DDPG, self).train()
        tf_sess = sess if sess else tf.get_default_session()

        batch_data_critic = self.replay_buffer.sample(
            batch_size=self.parameters('CRITIC_BATCH_SIZE')) if batch_data is None else batch_data
        assert isinstance(batch_data_critic, TransitionData)
        train_iter_critic = self.parameters("CRITIC_TRAIN_ITERATION") if not train_iter else train_iter

        critic_res = self._critic_train(batch_data_critic, train_iter_critic, tf_sess, update_target)

        batch_data_actor = self.replay_buffer.sample(
            batch_size=self.parameters('ACTOR_BATCH_SIZE')) if batch_data is None else batch_data
        assert isinstance(batch_data_actor, TransitionData)
        train_iter_actor = self.parameters("ACTOR_TRAIN_ITERATION") if not train_iter else train_iter

        actor_res = self._actor_train(batch_data_actor, train_iter_actor, tf_sess, update_target)

        return {**critic_res, **actor_res}

    def _critic_train(self, batch_data, train_iter, sess, update_target) -> dict:
        target_q = sess.run(
            self._target_critic_with_target_actor_output.q_tensor,
            feed_dict={
                self._target_critic_with_target_actor_output.state_input: batch_data.new_state_set,
                self.target_actor.state_input: batch_data.new_state_set
            }
        )
        average_loss = 0.0
        for _ in range(train_iter):
            loss, _ = sess.run(
                [self.critic_loss, self.critic_update_op],
                feed_dict={
                    self.target_q_input: target_q,
                    self.critic.state_input: batch_data.state_set,
                    self.critic.action_input: batch_data.action_set,
                    self.done_input: batch_data.done_set,
                    self.reward_input: batch_data.reward_set,
                    **self.parameters.return_tf_parameter_feed_dict()
                }
            )
            average_loss += loss

        if update_target is True:
            sess.run(self.target_critic_update_op)
        return dict(critic_average_loss=average_loss / train_iter)

    def _actor_train(self, batch_data, train_iter, sess, update_target) -> dict:
        average_loss = 0.0
        for _ in range(train_iter):
            loss, _ = sess.run(
                [self.actor_loss, self.actor_update_op],
                feed_dict={
                    self.actor.state_input: batch_data.state_set,
                    self._critic_with_actor_output.state_input: batch_data.state_set,
                    **self.parameters.return_tf_parameter_feed_dict()
                }
            )
            average_loss += loss
        if update_target is True:
            sess.run(self.target_actor_update_op)
        return dict(actor_average_loss=average_loss / train_iter)

    @register_counter_status_decorator(increment=1, key='test')
    def test(self, *arg, **kwargs) -> dict:
        return super().test(*arg, **kwargs)

    @typechecked
    def predict(self, obs: np.ndarray, sess=None, batch_flag: bool = False):
        tf_sess = sess if sess else tf.get_default_session()
        feed_dict = {
            self.state_input: make_batch(obs, original_shape=self.env_spec.obs_shape),
            **self.parameters.return_tf_parameter_feed_dict()
        }
        return self.actor.forward(obs=obs, sess=tf_sess, feed_dict=feed_dict)

    @typechecked
    def append_to_memory(self, samples: TransitionData):
        iter_samples = samples.return_generator()

        for obs0, obs1, action, reward, terminal1 in iter_samples:
            self.replay_buffer.append(obs0=obs0,
                                      obs1=obs1,
                                      action=action,
                                      reward=reward,
                                      terminal1=terminal1)

    def _setup_critic_loss(self):
        l1_l2 = tf_contrib.layers.l1_l2_regularizer(scale_l1=self.parameters('Q_NET_L1_NORM_SCALE'),
                                                    scale_l2=self.parameters('Q_NET_L2_NORM_SCALE'))
        loss = tf.reduce_sum((self.predict_q_value - self.critic.q_tensor) ** 2) + \
               tf_contrib.layers.apply_regularization(l1_l2, weights_list=self.critic.parameters('tf_var_list'))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters('CRITIC_LEARNING_RATE'))
        optimize_op = optimizer.minimize(loss=loss, var_list=self.critic.parameters('tf_var_list'))

        op = []
        for var, target_var in zip(self.critic.parameters('tf_var_list'),
                                   self.target_critic.parameters('tf_var_list')):
            ref_val = self.parameters('DECAY') * target_var + (1.0 - self.parameters('DECAY')) * var
            op.append(tf.assign(target_var, ref_val))

        return loss, optimize_op, op, optimizer

    def _set_up_actor_loss(self):
        loss = -tf.reduce_mean(self._critic_with_actor_output.q_tensor)
        grads = tf.gradients(loss, self.actor.parameters('tf_var_list'))
        if self.parameters('critic_clip_norm') is not None:
            grads = [tf.clip_by_norm(grad, clip_norm=self.parameters('critic_clip_norm')) for grad in grads]
        grads_var_pair = zip(grads, self.actor.parameters('tf_var_list'))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters('ACTOR_LEARNING_RATE'))
        optimize_op = optimizer.apply_gradients(grads_var_pair)
        op = []
        for var, target_var in zip(self.actor.parameters('tf_var_list'),
                                   self.target_actor.parameters('tf_var_list')):
            ref_val = self.parameters('DECAY') * target_var + (1.0 - self.parameters('DECAY')) * var
            op.append(tf.assign(target_var, ref_val))

        return loss, optimize_op, op, optimizer
