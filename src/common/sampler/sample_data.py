import numpy as np


# todo this module need to be tested

class SampleData(object):
    def __init__(self):
        self._state_set = []
        self._action_set = []
        self._reward_set = []
        self._done_set = []
        self._new_state_set = []
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0

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
        return np.array(self._state_set)

    @property
    def new_state_set(self):
        return np.array(self._new_state_set)

    @property
    def action_set(self):
        return np.array(self._action_set)

    @property
    def reward_set(self):
        return np.array(self._reward_set)

    @property
    def done_set(self):
        return np.array(self._done_set)


class TransitionData(SampleData):
    def __init__(self):
        super(TransitionData, self).__init__()


class TrajectoryData(SampleData):
    # todo implementation
    def __init__(self):
        super(TrajectoryData, self).__init__()
        self.trajectories = []
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0

    def reset(self):
        self.trajectories = []
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0

    def append(self, state, action, new_state, done, reward):
        self.cumulative_reward += reward

    def union(self, sample_data):
        pass
