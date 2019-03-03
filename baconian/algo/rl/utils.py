"""
Logging and Data Scaling Utilities

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)

            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n
        for i in range(len(self.vars)):
            if self.vars[i] == 0.0:
                self.vars[i] = 1e-6

        # print('scalar', self.means, self.vars)

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1 / (np.sqrt(self.vars) + 0.1) / 3, self.means


def _get_copy_arg_with_tf_reuse(obj, kwargs: dict):
    # kwargs = deepcopy(kwargs)
    if 'reuse' in kwargs:
        if kwargs['reuse'] is True:
            if 'name_scope' in kwargs and kwargs['name_scope'] != obj.name_scope:
                raise ValueError('If reuse, the name scope should be same instead of : {} and {}'.format(
                    kwargs['name_scope'], obj.name_scope))
            else:
                kwargs.update(name_scope=obj.name_scope)
        else:
            if 'name_scope' in kwargs and kwargs['name_scope'] == obj.name:
                raise ValueError(
                    'If not reuse, the name scope should be different instead of: {} and {}'.format(
                        kwargs['name_scope'], obj.name_scope))
    return kwargs
