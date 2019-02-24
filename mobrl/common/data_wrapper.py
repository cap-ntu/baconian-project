"""
A module for handle the data pre-processing including normalization, standardization,
"""
import numpy as np
import abc
from typeguard import typechecked
import tensorflow as tf


class DataWrapper(object):
    def __init__(self):
        self.unwrapped_data = None
        self.data = None

    def wrap(self, data):
        self.unwrapped_data = data
        # do sth to the data
        self.data = np.zeros([1])

    @abc.abstractmethod
    def __call__(self, input_data):
        raise NotImplementedError

    @property
    def unwrapped(self):
        return self.unwrapped_data


class Normalizer(DataWrapper):
    def __init__(self, mean, stddev, running_mean_std_flag=False):
        super(Normalizer, self).__init__()
        self.mean = mean
        self.stddev = stddev
        self.running_mean_std_flag = running_mean_std_flag

    @typechecked
    def wrap(self, data: (np.ndarray, tf.Tensor, tf.Variable,)):
        if isinstance(data, np.ndarray):
            pass

    def __call__(self, input_data):
        pass
