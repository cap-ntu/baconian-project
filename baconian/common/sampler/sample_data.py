from baconian.common.special import *
from baconian.core.core import EnvSpec
from copy import deepcopy


class SampleData(object):
    def __init__(self, env_spec: EnvSpec = None, obs_shape=None, action_shape=None):
        if env_spec is None and (obs_shape is None or action_shape is None):
            raise ValueError('At least env_spec or (obs_shape, action_shape) should be passed in')

        # self._state_set = []
        # self._action_set = []
        # self._reward_set = []
        # self._done_set = []
        # self._new_state_set = []
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0
        self.env_spec = env_spec
        self.obs_shape = env_spec.obs_shape if env_spec else obs_shape
        self.action_shape = env_spec.action_shape if env_spec else action_shape
        self._internal_data_dict = {
            'state_set': [[], self.obs_shape],
            'new_state_set': [[], self.obs_shape],
            'action_set': [[], self.action_shape],
            'reward_set': [[], [1]],
            'done_set': [[], [1]],
        }
        assert isinstance(self.obs_shape, (list, tuple))
        assert isinstance(self.action_shape, (list, tuple))

    def __len__(self):
        return len(self._internal_data_dict['state_set'][0])

    def reset(self):
        for key, data_set in self._internal_data_dict.items():
            # todo how to define the reset value
            self._internal_data_dict[key][0] = []
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0

    def append(self, *args, **kwargs):
        raise NotImplementedError

    def union(self, sample_data):
        assert isinstance(sample_data, type(self))
        self.cumulative_reward += sample_data.cumulative_reward
        self.step_count_per_episode += sample_data.step_count_per_episode
        for key, val in self._internal_data_dict.items():
            assert self._internal_data_dict[key][1] == sample_data._internal_data_dict[key][1]
            self._internal_data_dict[key][0] += deepcopy(sample_data._internal_data_dict[key][0])

    def get_copy(self):
        return deepcopy(self)

    def __call__(self, set_name, **kwargs):
        if set_name not in self._allowed_data_set_keys:
            raise ValueError('pass in set_name within {} '.format(self._allowed_data_set_keys))
        return make_batch(np.array(self._internal_data_dict[set_name][0]),
                          original_shape=self._internal_data_dict[set_name][1])

    def return_generator(self, batch_size=None, shuffle_flag=False):
        # todo this api need to be updated due to the append new dataset
        if batch_size or shuffle_flag:
            raise NotImplementedError
        else:
            for obs0, obs1, action, reward, terminal1 in zip(self._internal_data_dict['state_set'][0],
                                                             self._internal_data_dict['new_state_set'][0],
                                                             self._internal_data_dict['action_set'][0],
                                                             self._internal_data_dict['reward_set'][0],
                                                             self._internal_data_dict['done_set'][0]):
                yield obs0, obs1, action, reward, terminal1

    def append_new_set(self, name, data_set: (list, np.ndarray), shape: (tuple, list)):
        assert len(data_set) == len(self)
        assert len(np.array(data_set).shape) - 1 == len(shape)
        if len(shape) > 0:
            assert np.equal(np.array(data_set).shape[1:], shape).all()
        if isinstance(data_set, np.ndarray):
            data_set = data_set.tolist()
        shape = tuple(shape)
        self._internal_data_dict[name] = [data_set, shape]
        # self._allowed_data_set_keys.append(name)
        # self._register_data_set(name=name)

    def sample_batch(self, *args, **kwargs):
        raise NotImplementedError

    # todo in future version, use following api

    # def _register_data_set(self, name):
    #     self.__setattr__(name, )
    #     setattr(self, name, property(fget=self()))

    @property
    def _allowed_data_set_keys(self):
        return list(self._internal_data_dict.keys())

    @property
    def state_set(self):
        return self('state_set')

    @property
    def new_state_set(self):
        return self('new_state_set')

    @property
    def action_set(self):
        return self('action_set')

    @property
    def reward_set(self):
        return self('reward_set')

    @property
    def done_set(self):
        return self('done_set')

    def get_mean_of(self, set_name):
        return make_batch(np.array(self._internal_data_dict[set_name][0]),
                          original_shape=self._internal_data_dict[set_name][1]).mean()


class TransitionData(SampleData):
    def __init__(self, env_spec=None, obs_shape=None, action_shape=None):
        super(TransitionData, self).__init__(env_spec=env_spec, obs_shape=obs_shape, action_shape=action_shape)

    def sample_batch(self, batch_size, shuffle_flag=True, **kwargs) -> dict:
        if shuffle_flag is False:
            raise NotImplementedError
        total_num = len(self)
        id_index = np.random.randint(low=0, high=total_num, size=batch_size)
        batch_data = dict()
        for key in self._internal_data_dict.keys():
            batch_data[key] = self(key)[id_index]
        return batch_data

    def append(self, state: np.ndarray, action: np.ndarray, new_state: np.ndarray, done: bool, reward: float):
        # todo some type check should be here
        self._internal_data_dict['state_set'][0].append(state)
        self._internal_data_dict['new_state_set'][0].append(new_state)
        self._internal_data_dict['reward_set'][0].append(reward)
        self._internal_data_dict['done_set'][0].append(done)
        self._internal_data_dict['action_set'][0].append(action)
        self.cumulative_reward += reward


class TrajectoryData(SampleData):
    def __init__(self, env_spec=None, obs_shape=None, action_shape=None):
        super(TrajectoryData, self).__init__(env_spec=env_spec, obs_shape=obs_shape, action_shape=action_shape)
        self.trajectories = []

    def reset(self):
        self.trajectories = []

    @typechecked
    def append(self, transition_data: TransitionData):
        self.trajectories.append(transition_data.get_copy())

    @typechecked
    def union(self, sample_data):
        assert isinstance(sample_data, typechecked(self))
        self.trajectories += sample_data.trajectories

    def return_as_transition_data(self, shuffle_flag=False) -> TransitionData:
        if shuffle_flag:
            raise NotImplementedError
        transition_set = self.trajectories[0].get_copy()
        for i in range(1, len(self.trajectories)):
            transition_set.union(self.trajectories[i].get_copy())
        return transition_set

    def get_mean_of(self, set_name):
        tran = self.return_as_transition_data()
        return tran.get_mean_of(set_name)

    def __len__(self):
        total_len = 0
        for traj in self.trajectories:
            total_len += len(traj)
        return total_len
