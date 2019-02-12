import numpy as np
from typeguard import typechecked
from src.common.special import *
from src.envs.env_spec import EnvSpec
from copy import deepcopy


# todo this module need to be tested

class SampleData(object):
    def __init__(self, env_spec: EnvSpec = None, obs_shape=None, action_shape=None):
        if env_spec is None and (obs_shape is None or action_shape is None):
            raise ValueError('At least env_spec or (obs_shape, action_shape) should be passed in')

        self._state_set = []
        self._action_set = []
        self._reward_set = []
        self._done_set = []
        self._new_state_set = []
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0
        self.env_spec = env_spec
        self.obs_shape = env_spec.obs_shape if env_spec else obs_shape
        self.action_shape = env_spec.action_shape if env_spec else action_shape
        self._internal_data_dict = {
            'state_set': [self._state_set, self.obs_shape],
            'new_state_set': [self._new_state_set, self.obs_shape],
            'action_set': [self._action_set, self.action_shape],
            'reward_set': [self._reward_set, []],
            'done_set': [self._done_set, []],
        }
        assert isinstance(self.obs_shape, (list, tuple))
        assert isinstance(self.action_shape, (list, tuple))
        self._allowed_data_set_keys = ['state_set', 'action_set', 'new_state_set', 'reward_set', 'done_set']

    def __len__(self):
        return len(self._state_set)

    def reset(self):
        # self._state_set = []
        # self._action_set = []
        # self._reward_set = []
        # self._done_set = []
        # self._new_state_set = []
        for key, data_set in self._internal_data_dict.items():
            self._internal_data_dict[key][0] = []
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0

    def append(self, state: np.ndarray, action: np.ndarray, new_state: np.ndarray, done: bool, reward: (float,)):
        self._state_set.append(state)
        self._new_state_set.append(new_state)
        self._reward_set.append(reward)
        self._done_set.append(done)
        self._action_set.append(action)
        self.cumulative_reward += reward

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
            for obs0, obs1, action, reward, terminal1 in zip(self._state_set, self._new_state_set, self._action_set,
                                                             self._reward_set, self._done_set):
                yield obs0, obs1, action, reward, terminal1

    def append_new_set(self, name, data_set: (list, np.ndarray), shape: (tuple, list)):
        assert len(data_set) == len(self._action_set)
        assert len(np.array(data_set).shape) - 1 == len(shape)
        if len(shape) > 0:
            assert np.equal(np.array(data_set).shape[1:], shape)
        if isinstance(data_set, np.ndarray):
            data_set = data_set.tolist()
        shape = tuple(shape)
        self._internal_data_dict[name] = [data_set, shape]
        self._allowed_data_set_keys.append(name)

    def sample_batch(self, *args, **kwargs):
        raise NotImplementedError

    # todo the following api should be deprecated
    @property
    def state_set(self):
        return make_batch(np.array(self._state_set), original_shape=self.obs_shape)

    @property
    def new_state_set(self):
        return make_batch(np.array(self._new_state_set), self.obs_shape)

    @property
    def action_set(self):
        return make_batch(np.array(self._action_set), self.action_shape)

    @property
    def reward_set(self):
        return make_batch(np.array(self._reward_set), [1])

    @property
    def done_set(self):
        return make_batch(np.array(self._done_set), [1])


class TransitionData(SampleData):
    def __init__(self, env_spec=None, obs_shape=None, action_shape=None):
        super(TransitionData, self).__init__(env_spec=env_spec, obs_shape=obs_shape, action_shape=action_shape)

    def sample_batch(self, batch_size, shuffle_flag=True, **kwargs) -> dict:
        if shuffle_flag is False:
            raise NotImplementedError
        total_num = len(self._state_set)
        id_index = np.random.randint(low=0, high=total_num, size=batch_size)
        batch_data = dict()
        for key in self._internal_data_dict.keys():
            batch_data[key] = self(key)[id_index]
        return batch_data


class TrajectoryData(SampleData):
    # todo implementation
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

    def return_as_transition_data(self, shuffle_flag=False):
        if shuffle_flag:
            raise NotImplementedError
        transition_set = self.trajectories[0].get_copy()
        for i in range(1, len(self.trajectories)):
            transition_set.union(self.trajectories[i].make_copy())
        return transition_set
