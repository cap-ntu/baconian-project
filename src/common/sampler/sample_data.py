import numpy as np


class SampleData(object):
    pass


class TransitionData(SampleData):
    def __init__(self):
        self.state_set = []
        self.action_set = []
        self.reward_set = []
        self.done_set = []
        self.new_state_set = []
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0

    def reset(self):
        self.state_set = []
        self.action_set = []
        self.reward_set = []
        self.done_set = []
        self.new_state_set = []
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0

    def append(self, state: np.ndarray, action: np.ndarray, new_state: np.ndarray, done: bool, reward: (float,)):
        self.state_set.append(state)
        self.new_state_set.append(new_state)
        self.reward_set.append(reward)
        self.done_set.append(done)
        self.action_set.append(action)
        self.cumulative_reward += reward

    def union(self, sample_data):
        self.state_set += sample_data.state_set
        self.new_state_set += sample_data.new_state_set
        self.reward_set += sample_data.reward_set
        self.done_set += sample_data.done_set
        self.action_set += sample_data.action_set
        self.cumulative_reward += sample_data.cumulative_reward
        self.step_count_per_episode += sample_data.step_count_per_episode


class TrajectoryData(SampleData):
    # todo implementation
    def __init__(self):
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
