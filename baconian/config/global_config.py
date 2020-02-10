"""
The script to store some global configuration
"""

from typeguard import typechecked
import json_tricks as json
import tensorflow as tf
import os
from baconian.config.required_keys import SRC_UTIL_REQUIRED_KEYS
from baconian.common.error import *


class _SingletonDefaultGlobalConfig(object):
    DEFAULT_MAX_TF_SAVER_KEEP = 5
    DEFAULT_ALLOWED_EXCEPTION_OR_ERROR_LIST = (tf.errors.ResourceExhaustedError,)
    DEFAULT_BASIC_STATUS_LIST = ('TRAIN', 'TEST')
    DEFAULT_BASIC_INIT_STATUS = None

    # config required key list
    DEFAULT_DQN_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS, 'dqn.json')
    DEFAULT_CONSTANT_ACTION_POLICY_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                                    'constant_action_policy.json')

    DEFAULT_ALGO_DYNA_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                       'dyna.json')

    DEFAULT_ALGO_ME_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                     'me.json')
    DEFAULT_MODEL_FREE_PIPELINE_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                                 'model_free_pipeline.json')

    DEFAULT_MODEL_BASED_PIPELINE_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                                  'model_based_pipeline.json')

    DEFAULT_MPC_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                 'mpc.json')

    DEFAULT_DDPG_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                  'ddpg.json')
    DEFAULT_MADDPG_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS,
                                                    'maddpg.json')

    DEFAULT_PPO_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS, 'ppo.json')
    DEFAULT_AGENT_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS, 'agent.json')

    DEFAULT_EXPERIMENT_REQUIRED_KEY_LIST = os.path.join(SRC_UTIL_REQUIRED_KEYS, 'experiment.json')

    # LOGGING CONFIG

    DEFAULT_ALLOWED_LOG_FILE_TYPES = ('json', 'csv', 'h5py')
    DEFAULT_LOG_LEVEL = 'DEBUG'
    from baconian import ROOT_PATH
    DEFAULT_LOG_PATH = os.path.join(ROOT_PATH, 'test/tests/tmp_path')
    DEFAULT_MODEL_CHECKPOINT_PATH = os.path.join(DEFAULT_LOG_PATH, 'model_checkpoints')
    DEFAULT_LOG_CONFIG_DICT = dict()
    DEFAULT_LOG_USE_GLOBAL_MEMO_FLAG = True

    DEFAULT_LOGGING_FORMAT = '%(levelname)s:%(asctime)-15s: %(message)s'
    DEFAULT_WRITE_CONSOLE_LOG_TO_FILE_FLAG = True
    DEFAULT_CONSOLE_LOG_FILE_NAME = 'console.log'
    DEFAULT_CONSOLE_LOGGER_NAME = 'console_logger'
    DEFAULT_EXPERIMENT_END_POINT = dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=500,
                                        TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                        TOTAL_AGENT_UPDATE_COUNT=None)

    DEFAULT_TURN_OFF_GLOBAL_NAME_FLAG = False

    # For internal use
    SAMPLE_TYPE_SAMPLE_TRANSITION_DATA = 'transition_data'
    SAMPLE_TYPE_SAMPLE_TRAJECTORY_DATA = 'trajectory_data'

    def __init__(self):
        super().__setattr__('freeze_flag', False)

    def freeze(self):
        super().__setattr__('freeze_flag', True)

    def unfreeze(self):
        super().__setattr__('freeze_flag', False)

    @typechecked
    def set_new_config(self, config_dict: dict):
        for key, val in config_dict.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if attr is not None and not isinstance(val, type(attr)):
                    raise TypeError('Set the GlobalConfig.{} with type{}, instead of type {}'.format(key,
                                                                                                     type(
                                                                                                         attr).__name__,
                                                                                                     type(
                                                                                                         val).__name__))
                setattr(self, key, val)
            else:
                setattr(self, key, val)

    @typechecked
    def set_new_config_by_file(self, path_to_file: str):
        with open(path_to_file, 'r') as f:
            new_dict = json.load(f)
            self.set_new_config(new_dict)

    @typechecked
    def set(self, key: str, val):
        if self.freeze_flag is True:
            raise AttemptToChangeFreezeGlobalConfigError()
        if hasattr(self, key):
            attr = getattr(self, key)
            if attr is not None and not isinstance(val, type(attr)):
                raise TypeError('Set the GlobalConfig.{} with type{}, instead of type {}'.format(key,
                                                                                                 type(
                                                                                                     attr).__name__,
                                                                                                 type(
                                                                                                     val).__name__))
            setattr(self, key, val)
            if key == 'DEFAULT_LOG_PATH':
                self.set('DEFAULT_MODEL_CHECKPOINT_PATH', os.path.join(val, 'model_checkpoints'))
        else:
            setattr(self, key, val)

    def return_all_as_dict(self):
        return_dict = {}
        for key in dir(self):
            if key.isupper() is True or 'DEFAULT' in key:
                attr = getattr(self, key)
                try:
                    json.dumps(dict(key=attr))
                except TypeError as e:
                    attr = 'cannot be json dumped'
                return_dict[key] = attr
        return return_dict

    def __setattr__(self, key, value):
        if self.freeze_flag is True:
            raise AttemptToChangeFreezeGlobalConfigError("{} {}".format(key, value))
        else:
            super().__setattr__(key, value)


class GlobalConfig(object):
    only_instance = None

    def __new__(cls, *args, **kwargs):
        if GlobalConfig.only_instance is None:
            GlobalConfig.only_instance = _SingletonDefaultGlobalConfig()
        return GlobalConfig.only_instance
