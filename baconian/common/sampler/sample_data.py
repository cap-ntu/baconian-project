from baconian.common.special import *
from baconian.core.core import EnvSpec
from copy import deepcopy
import typeguard as tg
from baconian.common.error import *


class SampleData(object):
    def __init__(self, env_spec: EnvSpec = None, obs_shape=None, action_shape=None):
        if env_spec is None and (obs_shape is None or action_shape is None):
            raise ValueError('At least env_spec or (obs_shape, action_shape) should be passed in')
        self.env_spec = env_spec
        self.obs_shape = env_spec.obs_shape if env_spec else obs_shape
        self.action_shape = env_spec.action_shape if env_spec else action_shape

    def reset(self):
        raise NotImplementedError

    def append(self, *args, **kwargs):
        raise NotImplementedError

    def union(self, sample_data):
        raise NotImplementedError

    def get_copy(self):
        raise NotImplementedError

    def __call__(self, set_name, **kwargs):
        raise NotImplementedError

    def append_new_set(self, name, data_set: (list, np.ndarray), shape: (tuple, list)):
        raise NotImplementedError

    def sample_batch(self, *args, **kwargs):
        raise NotImplementedError

    def return_generator(self, *args, **kwargs):
        raise NotImplementedError

    def apply_transformation(self, set_name, func, *args, **kwargs):
        raise NotImplementedError

    def apply_op(self, set_name, func, *args, **kwargs):
        raise NotImplementedError


class TransitionData(SampleData):
    def __init__(self, env_spec: EnvSpec = None, obs_shape=None, action_shape=None):
        super(TransitionData, self).__init__(env_spec=env_spec, obs_shape=obs_shape, action_shape=action_shape)

        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0
        assert isinstance(self.obs_shape, (list, tuple))
        assert isinstance(self.action_shape, (list, tuple))
        self.obs_shape = list(self.obs_shape)
        self.action_shape = list(self.action_shape)

        self._internal_data_dict = {
            'state_set': [np.empty([0] + self.obs_shape), self.obs_shape],
            'new_state_set': [np.empty([0] + self.obs_shape), self.obs_shape],
            'action_set': [np.empty([0] + self.action_shape), self.action_shape],
            'reward_set': [np.empty([0]), []],
            'done_set': [np.empty([0], dtype=bool), []]
        }
        self.current_index = 0

    def __len__(self):
        return len(self._internal_data_dict['state_set'][0])

    def __call__(self, set_name, **kwargs):
        if set_name not in self._allowed_data_set_keys:
            raise ValueError('pass in set_name within {} '.format(self._allowed_data_set_keys))
        return make_batch(self._internal_data_dict[set_name][0],
                          original_shape=self._internal_data_dict[set_name][1])

    def reset(self):
        for key, data_set in self._internal_data_dict.items():
            self._internal_data_dict[key][0] = np.empty([0, *self._internal_data_dict[key][1]])
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0

    def append(self, state: np.ndarray, action: np.ndarray, new_state: np.ndarray, done: bool, reward: float):
        self._internal_data_dict['state_set'][0] = np.concatenate(
            (self._internal_data_dict['state_set'][0], np.reshape(state, [1] + self.obs_shape)), axis=0)
        self._internal_data_dict['new_state_set'][0] = np.concatenate(
            (self._internal_data_dict['new_state_set'][0], np.reshape(new_state, [1] + self.obs_shape)), axis=0)
        self._internal_data_dict['reward_set'][0] = np.concatenate(
            (self._internal_data_dict['reward_set'][0], np.reshape(reward, [1])), axis=0)
        self._internal_data_dict['done_set'][0] = np.concatenate(
            (self._internal_data_dict['done_set'][0], np.reshape(np.array(done, dtype=bool), [1])), axis=0)
        self._internal_data_dict['action_set'][0] = np.concatenate(
            (self._internal_data_dict['action_set'][0], np.reshape(action, [1] + self.action_shape)), axis=0)
        self.cumulative_reward += reward

    def union(self, sample_data):
        assert isinstance(sample_data, type(self))
        self.cumulative_reward += sample_data.cumulative_reward
        self.step_count_per_episode += sample_data.step_count_per_episode
        for key, val in self._internal_data_dict.items():
            assert self._internal_data_dict[key][1] == sample_data._internal_data_dict[key][1]
            self._internal_data_dict[key][0] = np.concatenate(
                (self._internal_data_dict[key][0], sample_data._internal_data_dict[key][0]), axis=0)

    def get_copy(self):
        obj = TransitionData(env_spec=self.env_spec, obs_shape=self.obs_shape, action_shape=self.action_shape)
        for key in self._internal_data_dict:
            obj._internal_data_dict[key] = deepcopy(self._internal_data_dict[key])
        return obj

    def append_new_set(self, name, data_set: (list, np.ndarray), shape: (tuple, list)):
        assert len(data_set) == len(self)
        assert len(np.array(data_set).shape) - 1 == len(shape)
        if len(shape) > 0:
            assert np.equal(np.array(data_set).shape[1:], shape).all()
        shape = tuple(shape)
        self._internal_data_dict[name] = [np.array(data_set), shape]

    def sample_batch(self, batch_size, shuffle_flag=True, **kwargs) -> dict:
        if shuffle_flag is False:
            raise NotImplementedError
        total_num = len(self)
        id_index = np.random.randint(low=0, high=total_num, size=batch_size)
        batch_data = dict()
        for key in self._internal_data_dict.keys():
            batch_data[key] = self(key)[id_index]
        return batch_data

    def get_mean_of(self, set_name):
        return self.apply_op(set_name=set_name, func=np.mean)

    def get_sum_of(self, set_name):
        return self.apply_op(set_name=set_name, func=np.sum)

    def apply_transformation(self, set_name, func, direct_apply=False, **func_kwargs):
        data = make_batch(self._internal_data_dict[set_name][0],
                          original_shape=self._internal_data_dict[set_name][1])
        transformed_data = make_batch(func(data, **func_kwargs),
                                      original_shape=self._internal_data_dict[set_name][1])
        if transformed_data.shape != data.shape:
            raise TransformationResultedToDifferentShapeError()
        elif direct_apply is True:
            self._internal_data_dict[set_name][0] = transformed_data
        return transformed_data

    def apply_op(self, set_name, func, **func_kwargs):
        data = make_batch(self._internal_data_dict[set_name][0],
                          original_shape=self._internal_data_dict[set_name][1])
        applied_op_data = np.array(func(data, **func_kwargs))
        return applied_op_data

    def shuffle(self, index: list = None):
        if not index:
            index = np.arange(len(self._internal_data_dict['state_set'][0]))
            np.random.shuffle(index)
        for key in self._internal_data_dict.keys():
            self._internal_data_dict[key][0] = self._internal_data_dict[key][0][index]

    def return_generator(self, batch_size=None, shuffle_flag=False, assigned_keys=None, infinite_run=False):
        if assigned_keys is None:
            assigned_keys = ('state_set', 'new_state_set', 'action_set', 'reward_set', 'done_set')
        # todo unit test should be tested, the dataset should not be shuffled, only the generator should be shuffled
        if shuffle_flag is True:
            self.shuffle()
        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError()
            start = 0
            if infinite_run is True:
                while True:
                    end = min(start + batch_size, len(self))
                    yield deepcopy(
                        [make_batch(self._internal_data_dict[key][0][start: end], self._internal_data_dict[key][1])
                         for key in assigned_keys])
                    start = end % len(self)
            else:
                while start < len(self):
                    end = min(start + batch_size, len(self))
                    yield deepcopy(
                        [make_batch(self._internal_data_dict[key][0][start: end], self._internal_data_dict[key][1])
                         for key in assigned_keys])
                    start = end
        else:
            start = 0
            if infinite_run is True:
                while True:
                    yield deepcopy([self._internal_data_dict[key][0][start] for key in assigned_keys])
                    start = (start + 1) % len(self)
            else:
                for i in range(len(self)):
                    yield deepcopy([self._internal_data_dict[key][0][i] for key in assigned_keys])

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


class TrajectoryData(SampleData):
    def __init__(self, env_spec=None, obs_shape=None, action_shape=None):
        super(TrajectoryData, self).__init__(env_spec=env_spec, obs_shape=obs_shape, action_shape=action_shape)
        self.trajectories = []

    def reset(self):
        self.trajectories = []

    def append(self, transition_data: TransitionData):
        self.trajectories.append(transition_data)

    def union(self, sample_data):
        if not isinstance(sample_data, type(self)):
            raise TypeError()
        self.trajectories += sample_data.trajectories

    def return_as_transition_data(self, shuffle_flag=False) -> TransitionData:
        transition_set = self.trajectories[0].get_copy()
        for i in range(1, len(self.trajectories)):
            transition_set.union(self.trajectories[i])
        if shuffle_flag is True:
            transition_set.shuffle()
        return transition_set

    def get_mean_of(self, set_name):
        tran = self.return_as_transition_data()
        return tran.get_mean_of(set_name)

    def get_sum_of(self, set_name):
        tran = self.return_as_transition_data()
        return tran.get_sum_of(set_name)

    def __len__(self):
        return len(self.trajectories)

    def get_copy(self):
        tmp_traj = TrajectoryData(env_spec=self.env_spec, obs_shape=self.obs_shape, action_shape=self.action_shape)
        for traj in self.trajectories:
            tmp_traj.append(transition_data=traj.get_copy())
        return tmp_traj

    def return_generator(self, batch_size=None, shuffle_flag=False):
        return self.return_as_transition_data(shuffle_flag=False).return_generator(batch_size=batch_size,
                                                                                   shuffle_flag=shuffle_flag)

    def apply_transformation(self, set_name, func, direct_apply=False, **func_kwargs):
        # TODO unit test
        for traj in self.trajectories:
            traj.apply_transformation(set_name, func, direct_apply, **func_kwargs)

    def apply_op(self, set_name, func, **func_kwargs):
        # TODO unit test
        res = []
        for traj in self.trajectories:
            res.append(traj.apply_op(set_name, func, **func_kwargs))
        return np.array(res)
