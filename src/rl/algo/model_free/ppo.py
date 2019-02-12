from src.envs.env_spec import EnvSpec
from src.rl.algo.algo import ModelFreeAlgo, OnPolicyAlgo
from src.config.dict_config import DictConfig
from typeguard import typechecked
import tensorflow as tf
import numpy as np
from src.common.sampler.sample_data import TrajectoryData, TransitionData
from src.tf.tf_parameters import TensorflowParameters
from src.config.global_config import GlobalConfig
from src.rl.policy.policy import StochasticPolicy
from src.rl.value_func.mlp_v_value import MLPVValueFunc
from src.tf.util import *
from src.common.misc import *
from src.rl.misc.sample_processor import SampleProcessor


class PPO(ModelFreeAlgo, OnPolicyAlgo):
    required_key_list = DictConfig.load_json(file_path=GlobalConfig.DEFAULT_PPO_REQUIRED_KEY_LIST)

    @typechecked
    def __init__(self, env_spec: EnvSpec,
                 stochastic_policy: StochasticPolicy,
                 config_or_config_dict: (DictConfig, dict),
                 # todo bug on mlp value function and its placeholder which is crushed with the dqn placeholder
                 value_func: MLPVValueFunc,
                 adaptive_learning_rate=False,
                 name='ppo'):
        super(PPO, self).__init__(env_spec, name)

        config = construct_dict_config(config_or_config_dict, self)
        self.config = config
        self.policy = stochastic_policy
        self.value_func = value_func
        self.adaptive_learning_rate = adaptive_learning_rate
        to_ph_parameter_dict = dict()
        self.trajectory_memory = TrajectoryData(env_spec=env_spec)
        self.transition_data_for_trajectory = TransitionData(env_spec=env_spec)
        self.value_func_train_data_buffer = None

        with tf.variable_scope(name):
            self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
            self.v_func_val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            # todo old_log_vars_ph and old_policy would limit the policy to be normal distribution

            # self.old_log_vars_ph = tf.placeholder(tf.float32, (self.env_spec.flat_action_dim,), 'old_log_vars')
            # self.old_means_ph = tf.placeholder(tf.float32, (None, self.env_spec.flat_action_dim), 'old_means')
            dist_info_list = self.policy.get_dist_info()
            self.old_dist_tensor = [
                (tf.placeholder(**dict(dtype=dist_info['dtype'],
                                       shape=dist_info['shape'],
                                       name=dist_info['name'])), dist_info['name']) for dist_info in
                dist_info_list
            ]
            self.old_policy = self.policy.make_copy(reuse=False,
                                                    name_scope='old_{}'.format(self.name),
                                                    distribution_tensors_tuple=tuple(self.old_dist_tensor))
            if adaptive_learning_rate is not False:
                to_ph_parameter_dict['policy_lr'] = tf.placeholder(shape=(), dtype=tf.float32,
                                                                   name='policy_lr')
                to_ph_parameter_dict['value_func_lr'] = tf.placeholder(shape=(), dtype=tf.float32,
                                                                       name='value_func_lr')
            to_ph_parameter_dict['beta'] = tf.placeholder(tf.float32, (), 'beta')
            to_ph_parameter_dict['eta'] = tf.placeholder(tf.float32, (), 'eta')
            to_ph_parameter_dict['kl_target'] = tf.placeholder(tf.float32, (), 'kl_target')

        self.parameters = TensorflowParameters(tf_var_list=[],
                                               rest_parameters=dict(
                                                   advantages_ph=self.advantages_ph,
                                                   v_func_val_ph=self.v_func_val_ph,
                                               ),
                                               to_ph_parameter_dict=to_ph_parameter_dict,
                                               name='ppo_param',
                                               auto_init=False,
                                               source_config=config,
                                               require_snapshot=False)
        with tf.variable_scope(name):
            with tf.variable_scope('train'):
                self.kl = tf.reduce_mean(self.old_policy.kl(other=self.policy))
                self.policy_loss, self.policy_optimizer, self.policy_update_op = self._setup_policy_loss()
                self.value_func_loss, self.value_func_optimizer, self.value_func_update_op = self._setup_value_func_loss()
        var_list = get_tf_collection_var_list(
            '{}/train'.format(name)) + self.policy_optimizer.variables() + self.value_func_optimizer.variables()
        self.parameters.set_tf_var_list(tf_var_list=sorted(list(set(var_list)), key=lambda x: x.name))

    def init(self):
        self.policy.init()
        self.value_func.init()
        sess = tf.get_default_session()
        sess.run(tf.variables_initializer(var_list=self.parameters('tf_var_list')))
        super().init()

    def train(self, trajectory_data: TrajectoryData = None, train_iter=None, sess=None) -> dict:
        super(PPO, self).train()
        if trajectory_data is None:
            trajectory_data = self.trajectory_memory

        tf_sess = sess if sess else tf.get_default_session()
        SampleProcessor.add_estimated_v_value(trajectory_data, value_func=self.value_func)
        SampleProcessor.add_discount_sum_reward(trajectory_data,
                                                gamma=self.parameters('gamma'))
        SampleProcessor.add_gae(trajectory_data,
                                gamma=self.parameters('gamma'),
                                name='advantage_set',
                                lam=self.parameters('lam'),
                                value_func=self.value_func)

        train_data = trajectory_data.return_as_transition_data(shuffle_flag=False)
        SampleProcessor.normalization(train_data, key='advantage_set')

        policy_res_dict = self._update_policy(train_data=train_data,
                                              train_iter=self.parameters('policy_train_iter'),
                                              sess=tf_sess)
        value_func_res_dict = self._update_value_func(train_data=train_data,
                                                      train_iter=self.parameters('value_func_train_iter'),
                                                      sess=tf_sess)
        return {
            **policy_res_dict,
            **value_func_res_dict
        }

    def test(self, *arg, **kwargs) -> dict:
        return super().test(*arg, **kwargs)

    @typechecked
    def predict(self, obs: np.ndarray, sess=None, batch_flag: bool = False):
        tf_sess = sess if sess else tf.get_default_session()
        return self.policy.forward(obs=obs, sess=tf_sess, feed_dict=self.parameters.return_tf_parameter_feed_dict())

    @typechecked
    def append_to_memory(self, samples: TransitionData):
        # todo how to make sure the data's time sequential
        iter_samples = samples.return_generator()
        for obs0, obs1, action, reward, terminal1 in iter_samples:
            self.transition_data_for_trajectory.append(state=obs0,
                                                       new_state=obs1,
                                                       action=action,
                                                       reward=reward,
                                                       done=terminal1)
            if terminal1 is True:
                self.trajectory_memory.append(self.transition_data_for_trajectory)
                self.transition_data_for_trajectory.reset()

    def _setup_policy_loss(self):
        """
        Code clip from pat-cody
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2

        See: https://arxiv.org/pdf/1707.02286.pdf
        """

        if self.parameters('clipping_range') is not None:
            print('setting up loss with clipping objective')
            pg_ratio = tf.exp(self.policy.log_prob() - self.old_policy.log_prob())
            clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - self.parameters('clipping_range')[0],
                                                1 + self.parameters('clipping_range')[1])
            surrogate_loss = tf.minimum(self.advantages_ph * pg_ratio,
                                        self.advantages_ph * clipped_pg_ratio)
            loss = -tf.reduce_mean(surrogate_loss)
        else:
            print('setting up loss with KL penalty')
            loss1 = -tf.reduce_mean(self.advantages_ph *
                                    tf.exp(self.policy.log_prob() - self.old_policy.log_prob()))
            loss2 = tf.reduce_mean(self.parameters('beta') * self.kl)
            loss3 = self.parameters('eta') * tf.square(
                tf.maximum(0.0, self.kl - 2.0 * self.parameters('kl_target')))
            loss = loss1 + loss2 + loss3
        optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters('policy_lr'))
        train_op = optimizer.minimize(loss, var_list=self.policy.parameters('tf_var_list'))
        return loss, optimizer, train_op

    def _setup_value_func_loss(self):
        loss = tf.reduce_mean(tf.square(self.value_func.v_tensor - self.v_func_val_ph))
        optimizer = tf.train.AdamOptimizer(self.parameters('value_func_lr'))
        train_op = optimizer.minimize(loss, var_list=self.value_func.parameters('tf_var_list'))
        return loss, optimizer, train_op

    def _update_policy(self, train_data: TransitionData, train_iter, sess):
        old_policy_feed_dict = dict()
        for tensor in self.old_dist_tensor:
            old_policy_feed_dict[tensor[0]] = sess.run(getattr(self.policy, tensor[1]),
                                                       feed_dict={
                                                           self.policy.parameters('state_input'): train_data(
                                                               'state_set'),
                                                           self.policy.parameters('action_input'): train_data(
                                                               'action_set'),

                                                           **self.parameters.return_tf_parameter_feed_dict()
                                                       })

        feed_dict = {
            self.policy.parameters('action_input'): train_data('action_set'),
            self.old_policy.parameters('action_input'): train_data('action_set'),
            self.policy.parameters('state_input'): train_data('state_set'),
            self.advantages_ph: train_data('advantage_set'),
            **self.parameters.return_tf_parameter_feed_dict(),
            **old_policy_feed_dict
        }
        average_loss, average_kl, average_entropy = 0.0, 0.0, 0.0
        total_epoch = 0
        for i in range(train_iter):
            loss, kl, entropy, _ = sess.run(
                [self.policy_loss, self.kl, tf.reduce_mean(self.policy.entropy()), self.policy_update_op],
                feed_dict=feed_dict)
            average_loss += loss
            average_kl += kl
            average_entropy += entropy
            total_epoch = i + 1
            if kl > self.parameters('kl_target', require_true_value=True) * 4:
                break
            # early stopping if D_KL diverges badly
        average_loss, average_kl, average_entropy = average_loss / total_epoch, average_kl / total_epoch, average_entropy / total_epoch

        # todo how to set parameters value
        # if kl > self.parameters('kl_target') * 2:  # servo beta to reach D_KL target
        #     self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
        #     if self.beta > 30 and self.lr_multiplier > 0.1:
        #         self.lr_multiplier /= 1.5
        # elif kl < self.kl_targ / 2:
        #     self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
        #     if self.beta < (1 / 30) and self.lr_multiplier < 10:
        #         self.lr_multiplier *= 1.5
        return dict(
            policy_average_loss=average_loss,
            policy_average_kl=average_kl,
            policy_average_entropy=average_entropy,
            policy_total_train_epoch=total_epoch
        )

    def _update_value_func(self, train_data: TransitionData, train_iter, sess):
        if self.value_func_train_data_buffer is None:
            self.value_func_train_data_buffer = train_data.get_copy()
        else:
            self.value_func_train_data_buffer.union(train_data)
        y_hat = self.value_func.forward(obs=train_data('state_set'))
        old_exp_var = 1 - np.var(train_data('advantage_set') - y_hat) / np.var(train_data('advantage_set'))
        num_batch = max(len(train_data) // self.parameters('value_func_train_batch_size'), 1)
        for i in range(train_iter):
            for j in range(num_batch):
                batch = self.value_func_train_data_buffer.sample_batch(
                    batch_size=self.parameters('value_func_train_batch_size'),
                    shuffle_flag=True)
                loss, _ = sess.run([self.value_func_loss, self.value_func_update_op],
                                   feed_dict={
                                       self.value_func.state_input: batch['state_set'],
                                       self.v_func_val_ph: batch['advantage_set'],
                                       **self.parameters.return_tf_parameter_feed_dict()
                                   })
        y_hat = self.value_func.forward(obs=train_data('state_set'))
        loss = np.mean(np.square(y_hat - train_data('advantage_set')))
        exp_var = 1 - np.var(train_data('advantage_set') - y_hat) / np.var(train_data('advantage_set'))
        return dict(
            value_func_loss=loss,
            value_func_policy_exp_var=exp_var,
            value_func_policy_old_exp_var=old_exp_var
        )
