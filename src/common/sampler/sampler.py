import numpy as np
import config as cfg
from src.core.basic import Basic
from src.core.config import Config


class SamplerData(object):
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

    def append(self, state, action, new_state, done, reward):
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


class Sampler(Basic):
    def __init__(self):
        super().__init__()
        self._test_data = SamplerData()
        self._real_data = SamplerData()
        self.step_count_per_episode = 0

    def init(self):
        self._test_data.reset()
        self._real_data.reset()

    def sample(self, env, algo, sample_count, store_flag=False):
        state = env.get_state(env)
        sample_record = SamplerData()
        for i in range(sample_count):
            action = algo.predict(state=state)
            new_state, re, done, info = env.step(action)
            if not isinstance(done, bool):
                if done[0] == 1:
                    done = True
                else:
                    done = False
            self.step_count_per_episode += 1
            if store_flag is True:
                algo.store_one_sample(state=state,
                                      action=action,
                                      next_state=new_state,
                                      reward=re,
                                      done=done)

            sample_record.append(state=state,
                                 action=action,
                                 reward=re,
                                 new_state=new_state,
                                 done=done)
            state = new_state
            if done is True:
                self.step_count_per_episode = 0
        return sample_record
