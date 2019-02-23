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
import logging
import abc
from mbrl.common.util.logger import global_logger


class Recorder(object):
    def __init__(self):
        self._obj_log = {}
        self._registered_log_attr_by_get_dict = {}
        global_logger.append_recorder(self)

    def append_to_obj_log(self, obj, obj_name: str, attr_name: str, status_info: dict, log_val):
        if obj not in self._obj_log:
            self._obj_log[obj] = {}
        if attr_name not in self._obj_log[obj]:
            self._obj_log[obj][attr_name] = []
        info = deepcopy(status_info)
        info['attr_name'] = deepcopy(attr_name)
        info['obj_name'] = deepcopy(obj_name)
        info['value'] = deepcopy(log_val)
        self._obj_log[obj][attr_name].append(info)

    def record(self):
        # todo how to call this function should be indicated
        self._record_by_getter()

    def _record_by_getter(self):
        for key, obj_dict in self._registered_log_attr_by_get_dict.items():
            for _, val in obj_dict.items():
                if val['get_method'] is None:
                    res = val['obj'].__getattribute__(val['attr_name'])
                else:
                    res = val['get_method'](val)
                self.append_to_obj_log(obj=val['obj'],
                                       attr_name=val['attr_name'],
                                       obj_name=val['obj_name'],
                                       log_val=res,
                                       status_info=val['obj'].get_status())

    @typechecked
    def register_logging_attribute_by_record(self, obj, obj_name: str, attr_name: str, static_flag: bool,
                                             get_method=None):
        """
        register an attribute that will be recorded periodically during training, duplicated registered will be ignored
        :param obj:
        :param obj_name:
        :param attr_name:
        :param static_flag:
        :param get_method:
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
                                                                     get_method=get_method,
                                                                     static_flag=static_flag)

    def reset(self):
        self._obj_log = {}
        self._registered_log_attr_by_get_dict = {}

    def flush(self):
        raise NotImplementedError


global_recorder = Recorder()


def reset_global_memo():
    global_recorder.reset()


def record_return_decorator(which_recorder: str = 'global'):
    def wrap(fn):
        def wrap_with_self(self, *args, **kwargs):
            obj = self
            obj_name = getattr(obj, 'name') if hasattr(obj, 'name') else str(obj.__name__)

            if which_recorder == 'global':
                recorder = global_recorder
            elif which_recorder == 'self':
                recorder = getattr(obj, 'recorder')
            else:
                raise ValueError('Not supported recorder indicator: {}, use {}'.format(which_recorder, 'gloabl, self'))
            if not hasattr(obj, 'get_status') or not callable(obj.get_status):
                raise ValueError('registered obj {} mush have callable method get_status()'.format(type(obj)))
            res = fn(self, *args, **kwargs)
            info = obj.get_status()
            if not isinstance(res, dict):
                raise TypeError('returned value by {} must be a dict in order to be recorded'.format(fn.__name__))
            for key, val in res.items():
                recorder.append_to_obj_log(obj=obj, attr_name=key, obj_name=obj_name, status_info=info,
                                           log_val=val)
            return res

        return wrap_with_self

    return wrap
