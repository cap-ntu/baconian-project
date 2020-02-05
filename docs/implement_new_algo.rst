How to implement a new algorithm
===================================


In this section, we will walk through the implementation of Deep Deterministic Policy Gradient (DDPG) algorithm, available at ``baconian/algo/ddpg.py``. It utilizes many functionalities
provided by the framework, which we will describe below.

- The ``ModelFreeAlgo`` and ``OffPolicyAlgo`` Classes

For the algorithms in Baconian project, we have writen many abstract classes to indicate
the characteristics of the algorithm, in ``baconian/algo/rl_algo.py``.
The ``DDPG`` class inherits from ``ModelFreeAlgo`` and ``OffPolicyAlgo`` classes' ``ModelFreeAlgo``, ``OffPolicyAlgo``
and other classes in ``baconian/algo/rl_algo.py`` inherit ``Algo`` class to categorize DRL algorithms.

.. literalinclude:: ../baconian/algo/rl_algo.py
    :language: python

Each new algorithm should implement the methods and attributes defined in ``Algo`` class (``baconian/algo/algo.py``).

.. literalinclude:: ../baconian/algo/algo.py
    :language: python

- The ``MultiPlaceholderInput`` Class

The algorithms in Baconian project are mostly implemented with TensorFlow, similar in the process of
saving and loading the parameters. Hence, parameters are stored in the format of TensorFlow variables by
``PlaceholderInput`` and ``MultiPlaceholderInput`` classes.

.. code-block:: python

    class DDPG(ModelFreeAlgo, OffPolicyAlgo, MultiPlaceholderInput):

        # ...

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

- Constructor

.. code-block:: python

    class DDPG(ModelFreeAlgo, OffPolicyAlgo, MultiPlaceholderInput):
        required_key_dict = DictConfig.load_json(file_path=GlobalConfig().DEFAULT_DDPG_REQUIRED_KEY_LIST)

        @typechecked()
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
            :param schedule_param_list:
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

            self.parameters = ParametersWithTensorflowVariable(tf_var_list=[],
                                                               rest_parameters=dict(),
                                                               to_scheduler_param_tuple=schedule_param_list,
                                                               name='ddpg_param',
                                                               source_config=config,
                                                               require_snapshot=False)
            """
            self.parameters contains all the parameters (variables) of the algorithm
            """
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


