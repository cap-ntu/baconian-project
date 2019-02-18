import logging
import os
import json
import numpy as np
from typeguard import typechecked
from mbrl.config.dict_config import DictConfig
from mbrl.common.misc import *
from mbrl.config.global_config import GlobalConfig
from copy import deepcopy
import decorators
from decorator import decorator

"""
Logger Module
1. a global console output file
2. each module/instance will have a single log file
3. support for tf related utility, tf model, tensorboard
4. support for different log file types
5. support for different level of logging
"""

# __all__ = ['Logger', 'count_call_times', 'global_logger', 'LoggingConfig']
_global_obj_log = {}
_registered_log_attr_by_get_dict = {}


class Logger(object):
    required_key_list = []

    def __init__(self, config_or_config_dict: (DictConfig, dict),
                 log_path, log_level=None, use_global_memo=True):
        self._log_dir = log_path
        self._config_file_log_dir = None
        self._loss_file_log_dir = None
        self._model_file_log_dir = None
        # todo debug mode
        # if os.path.exists(self._log_dir):
        #     raise FileExistsError('%s path is existed' % self._log_dir)
        self.logger_config = construct_dict_config(config_or_config_dict, obj=self)
        self.log_level = log_level
        self.use_global_memo = use_global_memo
        if use_global_memo is True:
            self._global_obj_log = globals()['_global_obj_log']
            self._registered_log_attr_by_get_dict = globals()['_registered_log_attr_by_get_dict']
            # self._registered_log_attr_by_decorator = globals()['_registered_log_attr_by_decorator']
        else:
            self._global_obj_log = {}
            self._registered_log_attr_by_get_dict = {}
            # self._registered_log_attr_by_decorator = {}

    @property
    def log_dir(self):
        if os.path.exists(self._log_dir) is False:
            os.makedirs(self._log_dir)
        return self._log_dir

    def out_to_json_file(self, file_path, content):
        with open(file_path, 'w') as f:
            # TODO how to modify this part
            for dict_i in content:
                for key, value in dict_i.items():
                    if isinstance(value, np.generic):
                        dict_i[key] = value.item()
            json.dump(content, fp=f, indent=4, sort_keys=True)

    def record(self):
        self._record_by_getter()

    def append_to_global_obj_log(self, obj, obj_name: str, attr_name: str, status_info: dict, log_val):
        if obj not in self._global_obj_log:
            self._global_obj_log[obj] = {}
        if attr_name not in self._global_obj_log[obj]:
            self._global_obj_log[obj][attr_name] = []
        info = deepcopy(status_info)
        info['attr_name'] = deepcopy(attr_name)
        info['obj_name'] = deepcopy(obj_name)
        info['value'] = deepcopy(log_val)
        self._global_obj_log[obj][attr_name].append(info)

    def _record_by_getter(self):
        for key, obj_dict in self._registered_log_attr_by_get_dict.items():
            for _, val in obj_dict.items():
                if val['get_method'] is None:
                    res = val['obj'].__getattribute__(val['attr_name'])
                else:
                    res = val['obj'].__getattribute__(val['get_method'])()
                self.append_to_global_obj_log(obj=val['obj'],
                                              attr_name=val['attr_name'],
                                              obj_name=val['obj_name'],
                                              log_val=res,
                                              status_info=val['obj'].get_status())

    @typechecked
    def register_logging_attribute_by_record(self, obj, obj_name: str, attr_name: str, static_flag: bool,
                                             get_method_name: str = None):
        """
        register an attribute that will be recorded periodically during training, duplicated registered will be ignored
        :param obj:
        :param obj_name:
        :param attr_name:
        :param static_flag:
        :param get_method_name:
        :return:
        """
        if not hasattr(obj, 'get_status') or not callable(obj.get_status):
            raise ValueError('registered obj {} mush have callable method get_status()'.format(type(obj)))
        if obj not in self._registered_log_attr_by_get_dict:
            self._registered_log_attr_by_get_dict[obj] = {}
        if attr_name in self._registered_log_attr_by_get_dict[obj]:
            return
        self._registered_log_attr_by_get_dict[obj][attr_name] = dict(obj=obj,
                                                                     obj_name=obj_name,
                                                                     attr_name=attr_name,
                                                                     get_method=get_method_name,
                                                                     static_flag=static_flag)

    # @typechecked
    # def _register_logging_attribute_by_decorator(self, obj, obj_name: str, attr_name: str):
    #     if not hasattr(obj, 'get_status') or not callable(obj.get_status):
    #         raise ValueError('registered obj {} mush have callable method get_status()'.format(type(obj)))
    #     if obj not in self._registered_log_file_dict:
    #         self._registered_log_file_dict[obj] = {}
    #     if attr_name in self._registered_log_file_dict[obj]:
    #         return
    #     self._registered_log_file_dict[obj][attr_name] = dict(obj=obj,
    #                                                           obj_name=obj_name,
    #                                                           attr_name=attr_name)


# @property
# def config_file_log_dir(self):
#     self._config_file_log_dir = os.path.join(self.log_dir, 'config')
#     if os.path.exists(self._config_file_log_dir) is False:
#         os.makedirs(self._config_file_log_dir)
#     return self._config_file_log_dir
#
# @property
# def loss_file_log_dir(self):
#     self._loss_file_log_dir = os.path.join(self.log_dir, 'loss')
#     if os.path.exists(self._loss_file_log_dir) is False:
#         os.makedirs(self._loss_file_log_dir)
#     return self._loss_file_log_dir
#
# @property
# def model_file_log_dir(self):
#     self._model_file_log_dir = os.path.join(self.log_dir, 'model')
#     if os.path.exists(self._model_file_log_dir) is False:
#         os.makedirs(self._model_file_log_dir)
#     return self._model_file_log_dir


global_logger = Logger(log_path=GlobalConfig.DEFAULT_LOG_PATH,
                       config_or_config_dict=GlobalConfig.DEFAULT_LOG_CONFIG_DICT,
                       use_global_memo=True,
                       log_level=GlobalConfig.DEFAULT_LOG_LEVEL)


class LoggingConfig(object):
    def __init__(self, log_file_type_list: list = GlobalConfig.DEFAULT_ALLOWED_LOG_FILE_TYPES,
                 log_level: int = GlobalConfig.DEFAULT_LOG_LEVEL):
        pass


@decorator
@typechecked
def count_call_times(func, logger: Logger = global_logger, *arg, **kwargs):
    obj = func.__self__


# @decorator
# def record_return(func, which_logger: str = 'global', self=None, *arg, **kwargs):
#     obj = self
#     obj_name = getattr(obj, 'name') if hasattr(obj, 'name') else str(obj.__name__)
#
#     if which_logger == 'global':
#         logger = global_logger
#     elif which_logger == 'self':
#         logger = getattr(obj, 'logger')
#     else:
#         raise ValueError('Not supported logger indicator: {}, use {}'.format(which_logger, 'gloabl, self'))
#     if not hasattr(obj, 'get_status') or not callable(obj.get_status):
#         raise ValueError('registered obj {} mush have callable method get_status()'.format(type(obj)))
#     res = func(self, *arg, **kwargs)
#     info = obj.get_status()
#     if not isinstance(res, dict):
#         raise TypeError('returned value by {} must be a dict in order to be logged'.format(func.__name__))
#     for key, val in res.items():
#         logger.append_to_global_obj_log(obj=obj, attr_name=key, obj_name=obj_name, status_info=info, log_val=val)


def record_return_decorator(which_logger: str):
    def wrap(fn):
        def wrap_with_self(self, *args, **kwargs):
            obj = self
            obj_name = getattr(obj, 'name') if hasattr(obj, 'name') else str(obj.__name__)

            if which_logger == 'global':
                logger = global_logger
            elif which_logger == 'self':
                logger = getattr(obj, 'logger')
            else:
                raise ValueError('Not supported logger indicator: {}, use {}'.format(which_logger, 'gloabl, self'))
            if not hasattr(obj, 'get_status') or not callable(obj.get_status):
                raise ValueError('registered obj {} mush have callable method get_status()'.format(type(obj)))
            res = fn(self, *args, **kwargs)
            info = obj.get_status()
            if not isinstance(res, dict):
                raise TypeError('returned value by {} must be a dict in order to be logged'.format(fn.__name__))
            for key, val in res.items():
                logger.append_to_global_obj_log(obj=obj, attr_name=key, obj_name=obj_name, status_info=info,
                                                log_val=val)
            return res

        return wrap_with_self

    return wrap


def reset_global_memo():
    globals()['_global_obj_log'] = {}
    globals()['_registered_log_attr_by_get_dict'] = {}
