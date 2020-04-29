from baconian.core.core import EnvSpec
from baconian.algo.rl_algo import ModelFreeAlgo, OffPolicyAlgo
from baconian.config.dict_config import DictConfig
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.algo.misc.replay_buffer import UniformRandomReplayBuffer, BaseReplayBuffer
import tensorflow as tf
from baconian.common.sampler.sample_data import TransitionData, TrajectoryData
from baconian.tf.tf_parameters import ParametersWithTensorflowVariable
from baconian.config.global_config import GlobalConfig
from baconian.common.special import *
from baconian.algo.policy import DeterministicMLPPolicy
from baconian.tf.util import *
from baconian.common.misc import *
from baconian.common.logging import record_return_decorator
from baconian.core.status import register_counter_info_to_status_decorator
from baconian.algo.misc.placeholder_input import MultiPlaceholderInput
from baconian.tf.util import clip_grad


class DDPG(ModelFreeAlgo, OffPolicyAlgo, MultiPlaceholderInput):
    required_key_dict = DictConfig.load_json(file_path=GlobalConfig().DEFAULT_DDPG_REQUIRED_KEY_LIST)

    @typechecked
    def __init__(self,
                 env_spec: EnvSpec,
                 config_or_config_dict: (DictConfig, dict),
                 value_func: MLPQValueFunction,
                 policy: DeterministicMLPPolicy,
                 schedule_param_list=None,
                 name='ddpg',
                 replay_buffer=None):
        """

        :param env_spec: environment specifications, like action apace or observation space
        :param config_or_config_dict: configuraion dictionary, like learning rate or decay, if any
        :param value_func: value function
        :param policy: agent policy
        :param schedule_param_list: schedule parameter list, if any  initla final function to schedule learning process
        :param name: name of algorithm class instance
        :param replay_buffer: replay buffer, if any
        """
        ModelFreeAlgo.__init__(self, env_spec=env_spec, name=name)
        config = construct_dict_config(config_or_config_dict, self)

        self.config = config
        self.actor = policy
        self.target_actor = self.actor.make_copy(name_scope='{}_target_actor'.format(self.name),
                                                 name='{}_target_actor'.format(self.name),
                                                 reuse=False)
        self.critic = value_func
        self.target_critic = self.critic.make_copy(name_scope='{}_target_critic'.format(self.name),
                                                   name='{}_target_critic'.format(self.name),
                                                   reuse=False)

        self.state_input = self.actor.state_input

        if replay_buffer:
            assert issubclass(replay_buffer, BaseReplayBuffer)
            self.replay_buffer = replay_buffer
        else:
            self.replay_buffer = UniformRandomReplayBuffer(limit=self.config('REPLAY_BUFFER_SIZE'),
                                                           action_shape=self.env_spec.action_shape,
                                                           observation_shape=self.env_spec.obs_shape)
        """
        self.parameters contains all the parameters (variables) of the algorithm
        """
        self.parameters = ParametersWithTensorflowVariable(tf_var_list=[],
                                                           rest_parameters=dict(),
                                                           to_scheduler_param_tuple=schedule_param_list,
                                                           name='ddpg_param',
                                                           source_config=config,
                                                           require_snapshot=False)
        self._critic_with_actor_output = self.critic.make_copy(reuse=True,
                                                               name='actor_input_{}'.format(self.critic.name),
                                                               state_input=self.state_input,
                                                               action_input=self.actor.action_tensor)
        self._target_critic_with_target_actor_output = self.target_critic.make_copy(reuse=True,
                                                                                    name='target_critic_with_target_actor_output_{}'.format(
                                                                                        self.critic.name),
                                                                                    action_input=self.target_actor.action_tensor)

        with tf.variable_scope(name):
            self.reward_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.next_state_input = tf.placeholder(shape=[None, self.env_spec.flat_obs_dim], dtype=tf.float32)
            self.done_input = tf.placeholder(shape=[None, 1], dtype=tf.bool)
            self.target_q_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            done = tf.cast(self.done_input, dtype=tf.float32)
            self.predict_q_value = (1. - done) * self.config('GAMMA') * self.target_q_input + self.reward_input
            with tf.variable_scope('train'):
                self.critic_loss, self.critic_update_op, self.target_critic_update_op, self.critic_optimizer, \
                self.critic_grads = self._setup_critic_loss()
                self.actor_loss, self.actor_update_op, self.target_actor_update_op, self.action_optimizer, \
                self.actor_grads = self._set_up_actor_loss()

        var_list = get_tf_collection_var_list(
            '{}/train'.format(name)) + self.critic_optimizer.variables() + self.action_optimizer.variables()
        self.parameters.set_tf_var_list(tf_var_list=sorted(list(set(var_list)), key=lambda x: x.name))
        MultiPlaceholderInput.__init__(self,
                                       sub_placeholder_input_list=[dict(obj=self.target_actor,
                                                                        attr_name='target_actor',
                                                                        ),
                                                                   dict(obj=self.actor,
                                                                        attr_name='actor'),
                                                                   dict(obj=self.critic,
                                                                        attr_name='critic'),
                                                                   dict(obj=self.target_critic,
                                                                        attr_name='target_critic')
                                                                   ],
                                       parameters=self.parameters)

    @register_counter_info_to_status_decorator(increment=1, info_key='init', under_status='INITED')
    def init(self, sess=None, source_obj=None):
        self.actor.init()
        self.critic.init()
        self.target_actor.init()
        self.target_critic.init(source_obj=self.critic)
        self.parameters.init()
        if source_obj:
            self.copy_from(source_obj)
        super().init()

    @record_return_decorator(which_recorder='self')
    @register_counter_info_to_status_decorator(increment=1, info_key='train', under_status='TRAIN')
    def train(self, batch_data=None, train_iter=None, sess=None, update_target=True) -> dict:
        super(DDPG, self).train()
        if isinstance(batch_data, TrajectoryData):
            batch_data = batch_data.return_as_transition_data(shuffle_flag=True)
        tf_sess = sess if sess else tf.get_default_session()
        train_iter = self.parameters("TRAIN_ITERATION") if not train_iter else train_iter
        average_critic_loss = 0.0
        average_actor_loss = 0.0
        for i in range(train_iter):
            train_batch = self.replay_buffer.sample(
                batch_size=self.parameters('BATCH_SIZE')) if batch_data is None else batch_data
            assert isinstance(train_batch, TransitionData)

            critic_loss, _ = self._critic_train(train_batch, tf_sess)

            actor_loss, _ = self._actor_train(train_batch, tf_sess)

            average_actor_loss += actor_loss
            average_critic_loss += critic_loss
        if update_target:
            tf_sess.run([self.target_actor_update_op, self.target_critic_update_op])
        return dict(average_actor_loss=average_actor_loss / train_iter,
                    average_critic_loss=average_critic_loss / train_iter)

    def _critic_train(self, batch_data, sess) -> ():
        target_q = sess.run(
            self._target_critic_with_target_actor_output.q_tensor,
            feed_dict={
                self._target_critic_with_target_actor_output.state_input: batch_data.new_state_set,
                self.target_actor.state_input: batch_data.new_state_set
            }
        )
        loss, _, grads = sess.run(
            [self.critic_loss, self.critic_update_op, self.critic_grads
             ],
            feed_dict={
                self.target_q_input: target_q,
                self.critic.state_input: batch_data.state_set,
                self.critic.action_input: batch_data.action_set,
                self.done_input: np.reshape(batch_data.done_set, [-1, 1]),
                self.reward_input: np.reshape(batch_data.reward_set, [-1, 1]),
                **self.parameters.return_tf_parameter_feed_dict()
            }
        )
        return loss, grads

    def _actor_train(self, batch_data, sess) -> ():
        target_q, loss, _, grads = sess.run(
            [self._critic_with_actor_output.q_tensor, self.actor_loss, self.actor_update_op, self.actor_grads],
            feed_dict={
                self.actor.state_input: batch_data.state_set,
                self._critic_with_actor_output.state_input: batch_data.state_set,
                **self.parameters.return_tf_parameter_feed_dict()
            }
        )
        return loss, grads

    @register_counter_info_to_status_decorator(increment=1, info_key='test', under_status='TEST')
    def test(self, *arg, **kwargs) -> dict:
        return super().test(*arg, **kwargs)

    def predict(self, obs: np.ndarray, sess=None, batch_flag: bool = False):
        tf_sess = sess if sess else tf.get_default_session()
        feed_dict = {
            self.state_input: make_batch(obs, original_shape=self.env_spec.obs_shape),
            **self.parameters.return_tf_parameter_feed_dict()
        }
        return self.actor.forward(obs=obs, sess=tf_sess, feed_dict=feed_dict)

    def append_to_memory(self, samples: TransitionData):
        iter_samples = samples.return_generator()

        for obs0, obs1, action, reward, terminal1 in iter_samples:
            self.replay_buffer.append(obs0=obs0,
                                      obs1=obs1,
                                      action=action,
                                      reward=reward,
                                      terminal1=terminal1)

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

    def _setup_critic_loss(self):
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.critic.name_scope)
        loss = tf.reduce_sum((self.predict_q_value - self.critic.q_tensor) ** 2)
        if len(reg_loss) > 0:
            loss += tf.reduce_sum(reg_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters('CRITIC_LEARNING_RATE'))
        grad_var_pair = optimizer.compute_gradients(loss=loss, var_list=self.critic.parameters('tf_var_list'))
        grads = [g[0] for g in grad_var_pair]
        if self.parameters('critic_clip_norm') is not None:
            grad_var_pair, grads = clip_grad(optimizer=optimizer,
                                             loss=loss,
                                             var_list=self.critic.parameters('tf_var_list'),
                                             clip_norm=self.parameters('critic_clip_norm'))
        optimize_op = optimizer.apply_gradients(grad_var_pair)
        op = []
        for var, target_var in zip(self.critic.parameters('tf_var_list'),
                                   self.target_critic.parameters('tf_var_list')):
            ref_val = self.parameters('DECAY') * target_var + (1.0 - self.parameters('DECAY')) * var
            op.append(tf.assign(target_var, ref_val))

        return loss, optimize_op, op, optimizer, grads

    def _set_up_actor_loss(self):
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.actor.name_scope)
        loss = -tf.reduce_mean(self._critic_with_actor_output.q_tensor)

        if len(reg_loss) > 0:
            loss += tf.reduce_sum(reg_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters('CRITIC_LEARNING_RATE'))
        grad_var_pair = optimizer.compute_gradients(loss=loss, var_list=self.actor.parameters('tf_var_list'))
        grads = [g[0] for g in grad_var_pair]
        if self.parameters('actor_clip_norm') is not None:
            grad_var_pair, grads = clip_grad(optimizer=optimizer,
                                             loss=loss,
                                             var_list=self.actor.parameters('tf_var_list'),
                                             clip_norm=self.parameters('critic_clip_norm'))
        optimize_op = optimizer.apply_gradients(grad_var_pair)
        op = []
        for var, target_var in zip(self.actor.parameters('tf_var_list'),
                                   self.target_actor.parameters('tf_var_list')):
            ref_val = self.parameters('DECAY') * target_var + (1.0 - self.parameters('DECAY')) * var
            op.append(tf.assign(target_var, ref_val))

        return loss, optimize_op, op, optimizer, grads

# todo identify API and their examples, limitations
