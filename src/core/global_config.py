"""
The script to store some global configuration
"""

import config as cfg
from typeguard import typechecked
import json
import tensorflow as tf
from gym.wrappers import TimeLimit, SkipWrapper, Monitor
from gym.core import Env as gym_env_cls
from src.envs import Env


class GlobalConfig(object):
    DEFAULT_MAX_TF_SAVER_KEEP = 20
    DEFAULT_CATCHED_EXCEPTION_OR_ERROR_LIST = (tf.errors.ResourceExhaustedError)
    # todo check this type list
    DEFAULT_ALLOWED_ENV_TYPE = (gym_env_cls, TimeLimit, SkipWrapper, Monitor, Env)

    @staticmethod
    @typechecked
    def set_new_config(config_dict: dict):
        for key, val in config_dict.items():
            if hasattr(GlobalConfig, key):
                # todo some type check here
                setattr(GlobalConfig, key, val)
            else:
                setattr(GlobalConfig, key, val)

    @staticmethod
    @typechecked
    def set_new_config_by_file(path_to_file: str):
        with open(path_to_file, 'r') as f:
            new_dict = json.load(f)
            GlobalConfig.set_new_config(new_dict)

