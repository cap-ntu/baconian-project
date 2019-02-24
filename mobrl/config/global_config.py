"""
The script to store some global configuration
"""

from typeguard import typechecked
import json
import tensorflow as tf
from gym.wrappers import TimeLimit, SkipWrapper, Monitor
import os
from mobrl.config.required_keys import SRC_UTIL_REQUIRED_KEYS


class GlobalConfig(object):
    DEFAULT_MAX_TF_SAVER_KEEP = 20
    DEFAULT_CATCHED_EXCEPTION_OR_ERROR_LIST = (tf.errors.ResourceExhaustedError,)
    # todo check this type list
    DEFAULT_ALLOWED_GYM_ENV_TYPE = (TimeLimit, SkipWrapper, Monitor)
    DEFAULT_BASIC_STATUS_LIST = ['TRAIN', 'TEST']
    DEFAULT_BASIC_INIT_STATUS = None

    # config required key list
    DEFAULT_DQN_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS, 'dqn.json')
    DEFAULT_ALGO_SAMPLE_WITH_DYNAMICS_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                                       'sample_with_dynamics.json')
    DEFAULT_MODEL_FREE_PIPELINE_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                                 'model_free_pipeline.json')

    DEFAULT_MODEL_BASED_PIPELINE_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                                  'model_based_pipeline.json')

    DEFAULT_MPC_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                 'mpc.json')

    DEFAULT_DDPG_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                  'ddpg.json')

    DEFAULT_PPO_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS, 'ppo.json')

    # LOGGING CONFIG

    DEFAULT_ALLOWED_LOG_FILE_TYPES = ('json', 'csv', 'h5py')
    # todo more detailed define on this
    DEFAULT_LOG_LEVEL = 'DEBUG'
    DEFAULT_LOG_PATH = '/home/dls/CAP/ModelBasedRLFramework/mobrl/test/tests/tmp_path'
    DEFAULT_LOG_CONFIG_DICT = dict()
    DEFAULT_LOG_USE_GLOBAL_MEMO_FLAG = True,

    DEFAULT_LOGGING_FORMAT = '%(asctime)-15s in path %(pathname)s: %(message)s'
    DEFAULT_WRITE_CONSOLE_LOG_TO_FILE_FLAG = True,
    DEFAULT_CONSOLE_LOG_FILE_NAME = 'console.log'

    @staticmethod
    @typechecked
    def set_new_config(config_dict: dict):
        for key, val in config_dict.items():
            if hasattr(GlobalConfig, key):
                attr = getattr(GlobalConfig, key)
                if attr is not None and not isinstance(val, type(attr)):
                    raise TypeError('Set the GlobalConfig.{} with type{}, instead of type {}'.format(key,
                                                                                                     type(
                                                                                                         attr).__name__,
                                                                                                     type(
                                                                                                         val).__name__))
                setattr(GlobalConfig, key, val)
            else:
                setattr(GlobalConfig, key, val)

    @staticmethod
    @typechecked
    def set_new_config_by_file(path_to_file: str):
        with open(path_to_file, 'r') as f:
            new_dict = json.load(f)
            GlobalConfig.set_new_config(new_dict)

    # For internal use
    SAMPLE_TYPE_SAMPLE_TRANSITION_DATA = 'transition_data'
    SAMPLE_TYPE_SAMPLE_TRAJECTORY_DATA = 'trajectory_data'
