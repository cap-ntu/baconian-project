"""
A scikit-learn liked module for handle the data pre-processing including normalization, standardization,
"""

import numpy as np
from baconian.common.error import *


class DataScaler(object):
    def __init__(self, dims):
        self.data_dims = dims

    def _check_scaler(self, scaler) -> bool:
        if len(scaler.shape) != 1 or scaler.shape[0] != self.data_dims:
            return False
        else:
            return True

    def _compute_stat_of_batch_data(self, data):
        data = np.array(data)
        if self._check_data(data) is False:
            raise ShapeNotCompatibleError("data shape is not compatible")
        else:
            return np.min(data, axis=0), np.max(data, axis=0), np.mean(data, axis=0), np.var(data, axis=0)

    def _check_data(self, data) -> bool:
        if len(data.shape) != 2 or data.shape[1] != self.data_dims:
            return False
        else:
            return True

    def process(self, data):
        raise NotImplementedError

    def inverse_process(self, data):
        raise NotImplementedError

    def update_scaler(self, data):
        pass


class IdenticalDataScaler(DataScaler):
    def __init__(self, dims):
        self.data_dims = dims

    def process(self, data):
        return data

    def inverse_process(self, data):
        return data


class MinMaxScaler(DataScaler):
    def __init__(self, dims: int, desired_range: tuple = None):
        super().__init__(dims)
        self._min = np.zeros([dims])
        self._max = np.ones([dims])

        if desired_range is None:
            self._desired_range = (np.zeros(dims), np.ones(dims))
        else:
            if len(desired_range) != 2 or self._check_scaler(np.array(desired_range[0])) is False or self._check_scaler(
                    np.array(desired_range[1])) is False:
                raise ShapeNotCompatibleError("desired value dims is not compatible with dims")
            self._desired_range = (np.array(desired_range[0]), np.array(desired_range[1]))

    def process(self, data):
        return (data - self._min) / (self._max - self._min) * (self._desired_range[1] - self._desired_range[0]) + \
               self._desired_range[0]

    def get_param(self):
        return dict(min=self._min.tolist(),
                    max=self._max.tolist(),
                    desired_range=np.array(self._desired_range).tolist())

    def set_param(self, min=None, max=None, desired_range=None):
        if self._check_scaler(min):
            self._min = min
        else:
            raise ShapeNotCompatibleError('the shape of min/max range is not as same as shape')
        if self._check_scaler(max):
            self._max = max
        else:
            raise ShapeNotCompatibleError('the shape of min/max range is not as same as shape')
        if len(desired_range) != 2 and self._check_scaler(np.array(desired_range[0])) is False or self._check_scaler(
                np.array(desired_range[1])) is False:
            raise ShapeNotCompatibleError("desired value dims is not compatible with dims")
        else:
            self._desired_range = (np.array(desired_range[0]), np.array(desired_range[1]))

    def inverse_process(self, data):
        if np.greater(np.array(data), self._desired_range[1]).any() or np.less(data, self._desired_range[0]).any():
            raise WrongValueRangeError('data for inverse process not in the range {} {}'.format(self._desired_range[0],
                                                                                                self._desired_range[1]))
        return (np.array(data) - self._desired_range[0]) / (self._desired_range[1] - self._desired_range[0]) * \
               (self._max - self._min) + self._min


class RunningMinMaxScaler(MinMaxScaler):
    """
    A scaler with running mean and max across all data updated into the scaler and scale the data to a desired range
    """

    def __init__(self, dims: int, desired_range: tuple = None, init_data: np.ndarray = None,
                 init_min: np.ndarray = None,
                 init_max: np.ndarray = None):
        super().__init__(dims=dims, desired_range=desired_range)
        if init_max is not None and init_max is not None:
            self._min = np.array(init_min)
            self._max = np.array(init_max)
        elif init_data is not None:
            self._min, self._max, _, _ = self._compute_stat_of_batch_data(init_data)
        if self._check_scaler(self._min) is False or self._check_scaler(self._max) is False:
            raise ShapeNotCompatibleError('the shape of min/max range is not as same as shape')

    def update_scaler(self, data):
        if len(np.array(data)) == 0:
            return
        if self._min is not None:
            self._min = np.minimum(np.min(data, axis=0), self._min)
        else:
            self._min = np.min(data, axis=0)
        if self._max is not None:
            self._max = np.maximum(np.max(data, axis=0), self._max)
        else:
            self._max = np.max(data, axis=0)


class BatchMinMaxScaler(MinMaxScaler):

    def process(self, data):
        if self._check_data(data) is False:
            raise ShapeNotCompatibleError("data is compatible with scaler, make sure only scale the box type data")
        self._min, self._max, _, _ = self._compute_stat_of_batch_data(data)
        return super().process(data)


class StandardScaler(DataScaler):
    def __init__(self, dims: int):
        super().__init__(dims)
        self._var = np.ones([dims])
        self._mean = np.zeros([dims])
        self._data_count = 0
        self._epsilon = 0.01

    def process(self, data):
        return (np.array(data) - self._mean) / (np.sqrt(self._var) + self._epsilon)

    def inverse_process(self, data):
        return np.array(data) * (np.sqrt(self._var) + self._epsilon) + self._mean

    def get_param(self):
        return dict(mean=self._mean.tolist(),
                    var=self._var.tolist())


class BatchStandardScaler(StandardScaler):
    def process(self, data):
        if self._check_data(data) is False:
            raise ShapeNotCompatibleError("data is compatible with scaler")
        _, _, self._mean, self._var = self._compute_stat_of_batch_data(data)
        return super().process(data)


class RunningStandardScaler(StandardScaler):
    """
    A scaler with running mean and variance across all data passed into the scaler and scale the data with zero mean and
    unit variance.
    """

    def __init__(self, dims: int, init_data: np.ndarray = None,
                 init_mean: np.ndarray = None,
                 init_var: np.ndarray = None,
                 init_mean_var_data_count=None):
        super().__init__(dims)

        if init_mean is not None and init_var is not None:
            self._mean = init_mean
            self._var = init_var
            self._data_count = init_mean_var_data_count
        elif init_data is not None:
            _, _, self._mean, self._var = self._compute_stat_of_batch_data(init_data)
            self._data_count = np.array(init_data).shape[0]

    def update_scaler(self, data):
        if len(np.array(data)) == 0:
            return
        if self._mean is None or self._var is None:
            _, _, self._mean, self._var = self._compute_stat_of_batch_data(data)
            self._data_count = data.shape[0]
        else:
            n = data.shape[0]
            new_data_var = np.var(data, axis=0)
            new_data_mean = np.mean(data, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self._mean * self._data_count) + (new_data_mean * n)) / (self._data_count + n)
            self._var = (((self._data_count * (self._var + np.square(self._mean))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self._data_count + n) -
                         np.square(new_means))
            self._var = np.maximum(0.0, self._var)
            self._mean = new_means
            self._data_count += n

    def set_param(self, mean=None, var=None):
        assert np.array(mean).shape == self._mean.shape, 'new mean shape is changed'
        assert np.array(var).shape == self._var.shape, 'new var shape is changed'
        self._mean = mean
        self._var = var
