import logging
import os
import json
import numpy as np
from typeguard import typechecked
from mobrl.config.dict_config import DictConfig
from mobrl.common.misc import *
from mobrl.config.global_config import GlobalConfig
from copy import deepcopy
import decorators
from decorator import decorator
import logging
import abc
from mobrl.common.util.logger import global_logger
from mobrl.core.basic import Basic


class Recorder(object):
    def __init__(self, flush_by_split_status=True):
        self._obj_log = {}
        self._registered_log_attr_by_get_dict = {}
        global_logger.append_recorder(self)
        self.flush_by_split_status = flush_by_split_status

    @typechecked
    def append_to_obj_log(self, obj: Basic, attr_name: str, status_info: dict, log_val):
        if obj not in self._obj_log:
            self._obj_log[obj] = {}
        if attr_name not in self._obj_log[obj]:
            self._obj_log[obj][attr_name] = []
        info = deepcopy(status_info)
        info['attr_name'] = deepcopy(attr_name)
        info['log_val'] = deepcopy(log_val)
        self._obj_log[obj][attr_name].append(info)

    def is_empty(self):
        return len(self._obj_log) == 0

    def record(self):
        # todo how to call this function should be indicated
        self._record_by_getter()

    @typechecked
    def register_logging_attribute_by_record(self, obj, attr_name: str, static_flag: bool,
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
                                                                     attr_name=attr_name,
                                                                     get_method=get_method,
                                                                     static_flag=static_flag)

    def _filter_by_main_status(self, clear_obj_log_flag=True):
        filtered_res = dict()
        for obj in self._obj_log:
            filtered_res[obj.name] = dict()
            status_list = obj.status_list
            for stat in status_list:
                filtered_res[obj.name][stat] = dict()
                for attr in self._obj_log[obj]:
                    filtered_res[obj.name][stat][attr] = []
            for attr in self._obj_log[obj]:
                for val_dict in self._obj_log[obj][attr]:
                    res = deepcopy(val_dict)
                    res.pop('status'), res.pop('attr_name')
                    filtered_res[obj.name][val_dict['status']][val_dict['attr_name']].append(res)
        if clear_obj_log_flag is True:
            self._obj_log = {}
        return filtered_res

    def get_obj_log_to_flush(self, clear_obj_log_flag) -> (dict, bool):
        if self.flush_by_split_status is True:
            return self._filter_by_main_status(clear_obj_log_flag), self.flush_by_split_status
        else:
            filtered_res = {}
            for obj in self._obj_log:
                filtered_res[obj.name] = dict()
                for attr in self._obj_log[obj]:
                    filtered_res[obj.name][attr] = []
                    for val_dict in self._obj_log[obj][attr]:
                        res = deepcopy(val_dict)
                        res.pop('attr_name')
                        filtered_res[obj.name][val_dict['attr_name']].append(res)
            self._obj_log = {}
            return filtered_res, self.flush_by_split_status

    def reset(self):
        self._obj_log = {}
        self._registered_log_attr_by_get_dict = {}

    def flush(self):
        global_logger.flush_recorder(recorder=self)

    def _record_by_getter(self):
        for key, obj_dict in self._registered_log_attr_by_get_dict.items():
            for _, val in obj_dict.items():
                if val['get_method'] is None:
                    res = val['obj'].__getattribute__(val['attr_name'])
                else:
                    res = val['get_method'](val)
                self.append_to_obj_log(obj=val['obj'],
                                       attr_name=val['attr_name'],
                                       log_val=res,
                                       status_info=val['obj'].get_status())


global_recorder = Recorder()


def reset_global_memo():
    global_recorder.reset()


def record_return_decorator(which_recorder: str = 'global'):
    def wrap(fn):
        def wrap_with_self(self, *args, **kwargs):
            obj = self

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
                recorder.append_to_obj_log(obj=obj, attr_name=key, status_info=info,
                                           log_val=val)
            return res

        return wrap_with_self

    return wrap
