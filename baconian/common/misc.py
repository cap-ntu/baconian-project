import numpy as np

__all__ = ['generate_n_actions_hot_code', 'repeat_ndarray', 'construct_dict_config']


def generate_n_actions_hot_code(n):
    res = np.arange(0, n)
    action = np.zeros([n, n])
    action[res, res] = 1
    return action


def repeat_ndarray(np_array: np.ndarray, repeats):
    np_array = np.expand_dims(np_array, axis=0)
    np_array = np.repeat(np_array, axis=0, repeats=repeats)
    return np_array


from baconian.config.dict_config import DictConfig


def construct_dict_config(config_or_config_dict, obj):
    if isinstance(config_or_config_dict, dict):
        return DictConfig(required_key_dict=obj.required_key_dict,
                          config_dict=config_or_config_dict,
                          cls_name=type(obj).__name__)
    elif isinstance(config_or_config_dict, dict):
        return config_or_config_dict
    else:
        raise TypeError('Type {} is not supported, use dict or Config'.format(type(config_or_config_dict).__name__))
