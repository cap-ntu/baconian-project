"""
The script to store some global configuration
"""

import config as cfg
from typeguard import typechecked
import json


class GlobalConfig(object):
    DEFAULT_MAX_TF_SAVER_KEEP = 20

    @staticmethod
    @typechecked
    def set_new_config(config_dict: dict):
        for key, val in config_dict.items():
            if hasattr(GlobalConfig, key):
                setattr(GlobalConfig, key, val)
            else:
                setattr(GlobalConfig, key, val)

    @staticmethod
    @typechecked
    def set_new_config_by_file(path_to_file: str):
        with open(path_to_file, 'r') as f:
            new_dict = json.load(f)
            GlobalConfig.set_new_config(new_dict)


if __name__ == '__main__':
    GlobalConfig.set_new_config(dict(DEFAULT_MAX_TF_SAVER_KEEP=10))
    print(GlobalConfig.DEFAULT_MAX_TF_SAVER_KEEP)
