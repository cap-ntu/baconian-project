from baconian.common.sampler.sample_data import TransitionData, TrajectoryData
from baconian.algo.value_func import ValueFunction
import scipy.signal
from baconian.common.special import *


def discount(x, gamma):
    """code clip from pat-cody"""
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


class SampleProcessor(object):

    @staticmethod
    def add_gae(data: TrajectoryData, gamma, lam, value_func: ValueFunction = None, name='advantage_set'):
        for traj in data.trajectories:
            # scale if gamma less than 1
            rewards = traj('reward_set') * (1 - gamma) if gamma < 0.999 else traj('reward_set')
            try:
                traj('v_value_set')
            except ValueError:
                if value_func is None:
                    raise ValueError('v_value_set did not existed, pass in value_func parameter to compute v_value_set')
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
    def add_discount_sum_reward(data: TrajectoryData, gamma, name='discount_set'):
        for traj in data.trajectories:
            # scale if gamma less than 1
            dis_set = traj('reward_set') * (1 - gamma) if gamma < 0.999 else traj('reward_set')
            # TODO add a unit test
            dis_reward_set = discount(np.reshape(dis_set, [-1, ]), gamma)
            traj.append_new_set(name=name, data_set=make_batch(dis_reward_set, original_shape=[]), shape=[])

    @staticmethod
    def add_estimated_v_value(data: (TrajectoryData, TransitionData), value_func: ValueFunction, name='v_value_set'):
        if isinstance(data, TrajectoryData):
            for path in data.trajectories:
                SampleProcessor._add_estimated_v_value(path, value_func, name)
        else:
            SampleProcessor._add_estimated_v_value(data, value_func, name)

    @staticmethod
    def _add_estimated_v_value(data: TransitionData, value_func: ValueFunction, name):
        v_set = value_func.forward(data.state_set)
        data.append_new_set(name=name, data_set=make_batch(np.array(v_set), original_shape=[]), shape=[])

    @staticmethod
    def normalization(data: (TransitionData, TrajectoryData), key, mean: np.ndarray = None, std_dev: np.ndarray = None):
        if isinstance(data, TransitionData):
            if mean is not None:
                assert mean.shape == data(key).shape[1:]
                assert std_dev.shape == data(key).shape[1:]
            else:
                mean = data(key).mean(axis=0)
                std_dev = data(key).std(axis=0)
            data.append_new_set(name=key, data_set=(data(key) - mean) / (std_dev + 1e-6), shape=data(key).shape[1:])
            return data
        else:
            # TODO add shape check
            mean = np.mean([d(key) for d in data.trajectories])
            std_dev = np.std([d(key) for d in data.trajectories])
            for d in data.trajectories:
                d.append_new_set(name=key, data_set=np.array((d(key) - mean) / (std_dev + 1e-6)),
                                 shape=d(key).shape[1:])
            return data
