import numpy as np


def generate_n_actions_hot_code(n):
    res = np.arange(0, n)
    action = np.zeros([n, n])
    action[res, res] = 1
    return action


def repeat_ndarray(np_array: np.ndarray, repeats):
    np_array = np.expand_dims(np_array, axis=0)
    np_array = np.repeat(np_array, axis=0, repeats=repeats)
    return np_array
