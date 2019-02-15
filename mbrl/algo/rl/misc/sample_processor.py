from mbrl.common.sampler.sample_data import TransitionData, TrajectoryData
import numpy as np
from mbrl.algo.rl.value_func import ValueFunction
from typeguard import typechecked
import scipy.signal
from mbrl.common.special import *


def discount(x, gamma):
    """code clip from pat-cody"""
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


class SampleProcessor(object):

    @staticmethod
    @typechecked
    def add_gae(data: TrajectoryData, gamma, lam, value_func: ValueFunction = None, name='advantage_set'):
        for traj in data.trajectories:
            rewards = traj('reward_set') * (1 - gamma) if gamma < 0.999 else traj('reward_set')
            try:
                traj('v_value_set')
            except ValueError:
                if value_func is None:
                    raise ValueError('v_value_set did not existed, pass in value_func parameter to compute')
                SampleProcessor.add_estimated_v_value(
                    data=traj,
                    value_func=value_func
                )
            finally:
                values = traj('v_value_set')
            # todo better way to handle shape error (no squeeze)
            tds = np.squeeze(rewards) - np.squeeze(values) + np.append(values[1:] * gamma, 0)
            advantages = discount(tds, gamma * lam)
            traj.append_new_set(name=name, data_set=make_batch(advantages, original_shape=[]), shape=[])

    @staticmethod
    @typechecked
    def add_discount_sum_reward(data: TrajectoryData, gamma, name='discount_set'):
        for traj in data.trajectories:
            dis_set = traj('reward_set') * (1 - gamma) if gamma < 0.999 else traj('reward_set')
            dis_set = discount(dis_set, gamma)
            traj.append_new_set(name=name, data_set=make_batch(dis_set, original_shape=[]), shape=[])

    @staticmethod
    @typechecked
    def add_estimated_v_value(data: (TrajectoryData, TransitionData), value_func: ValueFunction, name='v_value_set'):
        if isinstance(data, TrajectoryData):
            for path in data.trajectories:
                SampleProcessor._add_estimated_v_value(path, value_func, name)
        else:
            SampleProcessor._add_estimated_v_value(data, value_func, name)

    @staticmethod
    def _add_estimated_v_value(data: TransitionData, value_func: ValueFunction, name):
        data_iter = data.return_generator()
        v_set = []
        for obs0, _, _, _, _ in data_iter:
            v_set.append(value_func.forward(obs0))
        data.append_new_set(name=name, data_set=make_batch(np.array(v_set), original_shape=[]), shape=[])

    @staticmethod
    @typechecked
    def normalization(data: TransitionData, key, mean: np.ndarray = None, std_dev: np.ndarray = None):
        if mean is not None:
            assert mean.shape == data(key).shape[1:]
            assert std_dev.shape == data(key).shape[1:]
        else:
            mean = data(key).mean(axis=0)
            std_dev = data(key).std(axis=0)
        setattr(data, '_{}'.format(key), (data(key) - mean) / (std_dev + 1e-6))
