How to implement a new environment
=====================================



Similar to algorithms, environments in Baconian project also should implement the methods and attributes defined in
``Env`` class ``baconian/core/core.py``, inheriting gym Env class.

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


We use ``STATUS`` to record and control the status of an environment, ``register_counter_info_to_status_decorator`` is
a decorator that counts the times of initialization and reset of an environment.

.. code-block:: python

    def register_counter_info_to_status_decorator(increment, info_key, under_status: (str, tuple) = None,
                                                  ignore_wrong_status=False):
        def wrap(fn):
            if under_status:
                assert isinstance(under_status, (str, tuple))
                if isinstance(under_status, str):
                    final_st = tuple([under_status])
                else:
                    final_st = under_status

            else:
                final_st = (None,)

            @wraps(fn)
            def wrap_with_self(self, *args, **kwargs):
                # todo record() called in fn will lost the just appended info_key at the very first
                obj = self
                if not hasattr(obj, '_status') or not isinstance(getattr(obj, '_status'), StatusWithInfo):
                    raise ValueError(
                        ' the object {} does not not have attribute StatusWithInfo instance or hold wrong type of Status'.format(
                            obj))

                assert isinstance(getattr(obj, '_status'), StatusWithInfo)
                obj_status = getattr(obj, '_status')
                for st in final_st:
                    obj_status.append_new_info(info_key=info_key, init_value=0, under_status=st)
                res = fn(self, *args, **kwargs)
                for st in final_st:
                    if st and st != obj.get_status()['status'] and not ignore_wrong_status:
                        raise ValueError('register counter info under status: {} but got status {}'.format(st,
                                                                                                           obj.get_status()[
                                                                                                               'status']))
                obj_status.update_info(info_key=info_key, increment=increment,
                                       under_status=obj.get_status()['status'])
                return res

            return wrap_with_self

        return wrap

The class ``EnvSpec`` stores and regulates the environment specifications, e.g. data type of observation space and
action space in an environment.

.. code-block:: python

    class EnvSpec(object):
        @init_func_arg_record_decorator()
        @typechecked
        def __init__(self, obs_space: Space, action_space: Space):
            self._obs_space = obs_space
            self._action_space = action_space
            self.obs_shape = tuple(np.array(self.obs_space.sample()).shape)
            if len(self.obs_shape) == 0:
                self.obs_shape = (1,)
            self.action_shape = tuple(np.array(self.action_space.sample()).shape)
            if len(self.action_shape) == 0:
                self.action_shape = ()

        @property
        def obs_space(self):
            return self._obs_space

        @property
        def action_space(self):
            return self._action_space

        @property
        def flat_obs_dim(self) -> int:
            return int(flat_dim(self.obs_space))

        @property
        def flat_action_dim(self) -> int:
            return int(flat_dim(self.action_space))

        @staticmethod
        def flat(space: Space, obs_or_action: (np.ndarray, list)):
            return flatten(space, obs_or_action)

        def flat_action(self, action: (np.ndarray, list)):
            return flatten(self.action_space, action)

        def flat_obs(self, obs: (np.ndarray, list)):
            return flatten(self.obs_space, obs)