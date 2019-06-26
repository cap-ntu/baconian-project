How to implement a new algorithm/ environment / dynamics model
================================================================

(This page is under construction)
-------------------------------------


Implement new algorithms
---------------------------

In this section, we will walk through the implementation of Deep Deterministic Policy Gradient (DDPG) algorithm
, available at ``baconian/algo/ddpg.py``. It utilizes many functionalities
provided by the framework, which we describe below.

- The ``ModelFreeAlgo`` and ``OffPolicyAlgo`` Classes

The ``DDPG`` class inherits from ``ModelFreeAlgo`` and ``OffPolicyAlgo`` classes, which are abstract
class to indicate the characteristics of the algorithm. ``ModelFreeAlgo``, ``OffPolicyAlgo``
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
            # todo parameters
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


Implement new environment
---------------------------

Similar to algorithms, environments in Baconian project also should implement the methods and attributes defined in
``Env`` class (``baconian/core/core.py``).

.. code-block:: python

    class Env(gym.Env, Basic):
        """
        Abstract class for environment
        """
        key_list = ()
        STATUS_LIST = ('JUST_RESET', 'JUST_INITED', 'TRAIN', 'TEST', 'NOT_INIT')
        INIT_STATUS = 'NOT_INIT'

        @typechecked
        def __init__(self, name: str = 'env'):
            super(Env, self).__init__(status=StatusWithSubInfo(obj=self), name=name)
            self.action_space = None
            self.observation_space = None
            self.step_count = None
            self.recorder = Recorder()
            self._last_reset_point = 0
            self.total_step_count_fn = lambda: self._status.group_specific_info_key(info_key='step', group_way='sum')

        @register_counter_info_to_status_decorator(increment=1, info_key='step', under_status=('TRAIN', 'TEST'),
                                                   ignore_wrong_status=True)
        def step(self, action):
            pass

        @register_counter_info_to_status_decorator(increment=1, info_key='reset', under_status='JUST_RESET')
        def reset(self):
            self._status.set_status('JUST_RESET')
            self._last_reset_point = self.total_step_count_fn()

        @register_counter_info_to_status_decorator(increment=1, info_key='init', under_status='JUST_INITED')
        def init(self):
            self._status.set_status('JUST_INITED')

        def get_state(self):
            raise NotImplementedError

        def seed(self, seed=None):
            return self.unwrapped.seed(seed=seed)



Implement new dynamics model
-----------------------------
New dynamics model in Baconian project also implement the methods and attributes defined in
``DynamicsModel`` class (``baconian/algo/dynamics/dynamics_model.py``).

.. code-block:: python

    class DynamicsModel(Basic):
        STATUS_LIST = ('NOT_INIT', 'JUST_INITED')
        INIT_STATUS = 'NOT_INIT'

        def __init__(self, env_spec: EnvSpec, parameters: Parameters = None, init_state=None, name='dynamics_model',
                     state_input_scaler: DataScaler = None,
                     action_input_scaler: DataScaler = None,
                     state_output_scaler: DataScaler = None):
            """
            :param env_spec:
            :param parameters:
            :param init_state:
            :param name:
            """
            super().__init__(name=name)
            self.env_spec = env_spec
            self.state = init_state
            self.parameters = parameters
            self.state_input = None
            self.action_input = None
            self.new_state_output = None
            self.recorder = Recorder(flush_by_split_status=False)
            self._status = StatusWithSingleInfo(obj=self)
            self.state_input_scaler = state_input_scaler if state_input_scaler else IdenticalDataScaler(
                dims=env_spec.flat_obs_dim)
            self.action_input_scaler = action_input_scaler if action_input_scaler else IdenticalDataScaler(
                dims=env_spec.flat_action_dim)
            self.state_output_scaler = state_output_scaler if state_output_scaler else IdenticalDataScaler(
                dims=env_spec.flat_obs_dim)

        def init(self, *args, **kwargs):
            self.set_status('JUST_INITED')
            self.state = self.env_spec.obs_space.sample()

        @register_counter_info_to_status_decorator(increment=1, info_key='step_counter')
        def step(self, action: np.ndarray, state=None, allow_clip=False, **kwargs_for_transit):
            """
            State transition function (only support one sample transition instead of batch data)

            :param action: action to be taken
            :param state: current state, if None, will use stored state (saved from last transition)
            :param allow_clip: boolean, if True, will clip the output to fit it bound, if False, will not, and if the output
                is out bound, will throw an error
            :param kwargs_for_transit: extra kwargs for calling the _state_transit, this is typically related to the specific
                mode you used
            :return:
            """
            state = np.array(state).reshape(self.env_spec.obs_shape) if state is not None else self.state
            action = action.reshape(self.env_spec.action_shape)
            if allow_clip is True:
                if state is not None:
                        state = self.env_spec.obs_space.clip(state)
                action = self.env_spec.action_space.clip(action)
            if self.env_spec.action_space.contains(action) is False:
                raise StateOrActionOutOfBoundError(
                    'action {} out of bound of {}'.format(action, self.env_spec.action_space.bound()))
            if self.env_spec.obs_space.contains(state) is False:
                raise StateOrActionOutOfBoundError(
                    'state {} out of bound of {}'.format(state, self.env_spec.obs_space.bound()))
            new_state = self._state_transit(state=state, action=self.env_spec.flat_action(action),
                                            **kwargs_for_transit)
            if allow_clip is True:
                new_state = self.env_spec.obs_space.clip(new_state)
            if self.env_spec.obs_space.contains(new_state) is False:
                raise StateOrActionOutOfBoundError(
                    'new state {} out of bound of {}'.format(new_state, self.env_spec.obs_space.bound()))
            self.state = new_state
            return new_state

        @abc.abstractmethod
        def _state_transit(self, state, action, **kwargs) -> np.ndarray:
            raise NotImplementedError

        def copy_from(self, obj) -> bool:
            if not isinstance(obj, type(self)):
                raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
            return True

        def make_copy(self):
            raise NotImplementedError

        def reset_state(self, state=None):
            if state is not None:
                assert self.env_spec.obs_space.contains(state)
                self.state = state
            else:
                self.state = self.env_spec.obs_space.sample()

        def return_as_env(self) -> Env:
            return DynamicsEnvWrapper(dynamics=self,
                                      name=self._name + '_env')

Similar to algorithms, dynamics models are categorized in ``baconian/algo/dynamics/dynamics_model.py``,
such as ``GlobalDynamicsModel``.
