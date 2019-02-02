import numpy as np
from typeguard import typechecked
from src.misc.special import *
from src.envs.env_spec import EnvSpec


# todo this module need to be tested

class SampleData(object):
    def __init__(self, env_spec: EnvSpec = None, obs_shape=None, action_shape=None):
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
        assert isinstance(self.obs_shape, (list, tuple))
        assert isinstance(self.action_shape, (list, tuple))

    def reset(self):
        self._state_set = []
        self._action_set = []
        self._reward_set = []
        self._done_set = []
        self._new_state_set = []
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
        self._state_set += sample_data._state_set
        self._new_state_set += sample_data._new_state_set
        self._reward_set += sample_data._reward_set
        self._done_set += sample_data._done_set
        self._action_set += sample_data._action_set
        self.cumulative_reward += sample_data._cumulative_reward
        self.step_count_per_episode += sample_data.step_count_per_episode

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

    def return_generator(self, batch_size=None, shuffle_flag=False):
        if batch_size or shuffle_flag:
            raise NotImplementedError
        else:
            for obs0, obs1, action, reward, terminal1 in zip(self._state_set, self._new_state_set, self._action_set,
                                                             self._reward_set, self._done_set):
                yield obs0, obs1, action, reward, terminal1


class TransitionData(SampleData):
    def __init__(self, env_spec=None, obs_shape=None, action_shape=None):
        super(TransitionData, self).__init__(env_spec=env_spec, obs_shape=obs_shape, action_shape=action_shape)


class TrajectoryData(SampleData):
    # todo implementation
    def __init__(self, env_spec=None, obs_shape=None, action_shape=None):
        super(TrajectoryData, self).__init__(env_spec=env_spec, obs_shape=obs_shape, action_shape=action_shape)
        self.trajectories = []

    def reset(self):
        self.trajectories = []

    @typechecked
    def append(self, transition_data: TransitionData):
        self.trajectories.append(self._process_trajecotry(transition_data))

    def union(self, sample_data):
        raise NotImplementedError

    def _process_trajectory(self, trajectory_data) -> TransitionData:
        raise NotImplementedError
