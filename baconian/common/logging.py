import abc
import logging
import os

from baconian.common.misc import construct_dict_config
from baconian.common import files as files
from baconian.core.global_var import get_all
from baconian.config.global_config import GlobalConfig
from functools import wraps
from baconian.common.error import *


class BaseLogger(object):
    required_key_dict = ()

    def __init__(self):
        self.inited_flag = False

    @abc.abstractmethod
    def close(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def init(self, *args, **kwargs):
        raise NotImplementedError


class _SingletonConsoleLogger(BaseLogger):
    """
    A private class that should never be instanced, it is used to implement the singleton design pattern for
    ConsoleLogger
    """
    ALLOWED_LOG_LEVEL = ('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET')
    ALLOWED_PRINT_TYPE = ('info', 'warning', 'debug', 'critical', 'log', 'critical', 'error')

    def __init__(self):
        super(_SingletonConsoleLogger, self).__init__()
        self.name = None
        self.logger = None

    def init(self, to_file_flag, level: str, to_file_name: str = None, logger_name: str = 'console_logger'):
        if self.inited_flag is True:
            return
        self.name = logger_name
        if level not in self.ALLOWED_LOG_LEVEL:
            raise ValueError('Wrong log level use {} instead'.format(self.ALLOWED_LOG_LEVEL))
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, level))

        for handler in self.logger.root.handlers[:] + self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            self.logger.root.removeHandler(handler)

        self.logger.addHandler(logging.StreamHandler())
        if to_file_flag is True:
            self.logger.addHandler(logging.FileHandler(filename=to_file_name))
        for handler in self.logger.root.handlers[:] + self.logger.handlers[:]:
            handler.setFormatter(fmt=logging.Formatter(fmt=GlobalConfig().DEFAULT_LOGGING_FORMAT))
            handler.setLevel(getattr(logging, level))

        self.inited_flag = True

    def print(self, p_type: str, p_str: str, *arg, **kwargs):
        if p_type not in self.ALLOWED_PRINT_TYPE:
            raise ValueError('use print type from {}'.format(self.ALLOWED_PRINT_TYPE))
        getattr(self.logger, p_type)(p_str, *arg, **kwargs)
        self.flush()

    def close(self):
        self.flush()
        for handler in self.logger.root.handlers[:] + self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
            self.logger.root.removeHandler(handler)

    def reset(self):
        self.close()
        self.inited_flag = False

    def flush(self):
        for handler in self.logger.root.handlers[:] + self.logger.handlers[:]:
            handler.flush()


class _SingletonLogger(BaseLogger):
    """
    A private class that should never be instanced, it is used to implement the singleton design pattern for Logger
    """

    def __init__(self):
        super(_SingletonLogger, self).__init__()
        self._registered_recorders = []
        self._log_dir = None
        self._config_file_log_dir = None
        self._record_file_log_dir = None
        self.logger_config = None
        self.log_level = None

    def init(self, config_or_config_dict,
             log_path, log_level=None, **kwargs):
        if self.inited_flag:
            return
        self._log_dir = log_path
        # if os.path.exists(self._log_dir):
        #     raise FileExistsError('%s path is existed' % self._log_dir)
        self._config_file_log_dir = os.path.join(self._log_dir, 'config')
        self._record_file_log_dir = os.path.join(self._log_dir, 'record')

        self.logger_config = construct_dict_config(config_or_config_dict, obj=self)
        self.log_level = log_level
        self.inited_flag = True

    @property
    def log_dir(self):
        if os.path.exists(self._log_dir) is False:
            os.makedirs(self._log_dir)
        return self._log_dir

    def flush_recorder(self, recorder=None):
        if not recorder:
            for re in self._registered_recorders:
                self._flush(re)
        else:
            self._flush(recorder)

    def close(self):
        self._save_all_obj_final_status()
        self.flush_recorder()
        self._registered_recorders = []

    def append_recorder(self, recorder):
        self._registered_recorders.append(recorder)

    def reset(self):
        self.close()
        for re in self._registered_recorders:
            re.reset()
        self._registered_recorders = []
        self.inited_flag = False

    def _flush(self, recorder):
        if recorder.is_empty():
            return
        log_dict, by_status_flag = recorder.get_obj_log_to_flush(clear_obj_log_flag=True)
        for obj_name, obj_log_dict in log_dict.items():
            if by_status_flag is True:
                for status, status_log_dict in obj_log_dict.items():
                    ConsoleLogger().print('info', 'save {}, with status: {} log into {}'.format(str(obj_name),
                                                                                                str(status),
                                                                                                os.path.join(
                                                                                                    self._record_file_log_dir,
                                                                                                    str(obj_name),
                                                                                                    str(status))))
                    self.out_to_file(
                        file_path=os.path.join(self._record_file_log_dir, str(obj_name), str(status)),
                        content=status_log_dict,
                        file_name='log.json')
            else:
                ConsoleLogger().print('info', 'save {} log into {}'.format(str(obj_name),
                                                                           os.path.join(
                                                                               self._record_file_log_dir,
                                                                               str(obj_name))))
                self.out_to_file(file_path=os.path.join(self._record_file_log_dir, str(obj_name)),
                                 content=obj_log_dict,
                                 file_name='log.json')

    def _save_all_obj_final_status(self):
        final_status = dict()
        for obj_name, obj in get_all()['_global_name_dict'].items():
            if hasattr(obj, 'get_status') and callable(getattr(obj, 'get_status')):
                tmp_dict = dict()
                tmp_dict[obj_name] = dict()
                for st in obj.STATUS_LIST:
                    obj.set_status(st)
                    tmp_dict[obj_name][st] = obj.get_status()
                final_status = {**final_status, **tmp_dict}
        ConsoleLogger().print('info', 'save final_status into {}'.format(os.path.join(
            self._record_file_log_dir)))
        self.out_to_file(file_path=os.path.join(self._record_file_log_dir),
                         content=final_status,
                         force_new=True,
                         file_name='final_status.json')
        ConsoleLogger().print('info', 'save global_config into {}'.format(os.path.join(
            self._record_file_log_dir)))
        self.out_to_file(file_path=os.path.join(self._record_file_log_dir),
                         content=GlobalConfig().return_all_as_dict(),
                         force_new=True,
                         file_name='global_config.json')

    @staticmethod
    def out_to_file(file_path: str, content: (tuple, list, dict), file_name: str, force_new=False):

        if len(content) == 0:
            return
        if force_new is True:
            mode = 'w'
        else:
            mode = 'a'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            mode = 'w'
        try:
            f = open(os.path.join(file_path, file_name), mode)
        except FileNotFoundError:
            f = open(os.path.join(file_path, file_name), 'w')
        files.save_to_json(content, fp=f)


class Logger(object):
    only_instance = None

    def __new__(cls, *args, **kwargs):
        if Logger.only_instance is None:
            Logger.only_instance = _SingletonLogger()
        return Logger.only_instance


class ConsoleLogger(object):
    only_instance = None

    def __new__(cls, *args, **kwargs):
        if not ConsoleLogger.only_instance:
            ConsoleLogger.only_instance = _SingletonConsoleLogger()
        return ConsoleLogger.only_instance


class Recorder(object):
    def __init__(self, flush_by_split_status=True, default_obj=None):
        self._obj_log = {}
        self._registered_log_attr_by_get_dict = {}
        Logger().append_recorder(self)
        self.flush_by_split_status = flush_by_split_status
        self._default_obj = default_obj

    def append_to_obj_log(self, obj, attr_name: str, status_info: dict, log_val):
        assert hasattr(obj, 'name')
        if obj not in self._obj_log:
            self._obj_log[obj] = {}
        if attr_name not in self._obj_log[obj]:
            self._obj_log[obj][attr_name] = []
        self._obj_log[obj][attr_name].append(dict(**status_info, attr_name=attr_name, log_val=log_val))

    def get_log(self, attr_name: str, filter_by_status: dict = None, obj=None):
        if obj is None:
            obj = self._default_obj
        if obj not in self._obj_log:
            raise LogItemNotExisted('object {} has no records in this recorder'.format(obj))
        if attr_name not in self._obj_log[obj]:
            raise LogItemNotExisted('no log item {} found at object {} recorder'.format(attr_name, obj))
        record = self._obj_log[obj][attr_name]
        if filter_by_status is not None:
            # TODO reduce the time complexity of the code snippet
            filtered_record = []
            for r in record:
                not_equal_flag = False
                for key in filter_by_status.keys():
                    if key in r and r[key] != filter_by_status[key]:
                        not_equal_flag = True
                if not not_equal_flag:
                    filtered_record.append(r)
            return filtered_record
        else:
            return record

    def is_empty(self):
        return len(self._obj_log) == 0

    def record(self):
        self._record_by_getter()

    def register_logging_attribute_by_record(self, obj, attr_name: str, static_flag: bool,
                                             get_method=None):
        """
        register an attribute that will be recorded periodically during training, duplicated registered will be ignored

        :param obj:
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
                    filtered_res[obj.name][val_dict['status']][val_dict['attr_name']].append(val_dict)
                    filtered_res[obj.name][val_dict['status']][attr][-1].pop('attr_name')
        if clear_obj_log_flag is True:
            del self._obj_log
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
                        val_dict.pop('attr_name')
                        filtered_res[obj.name][attr].append(val_dict)
            del self._obj_log
            self._obj_log = {}
            return filtered_res, self.flush_by_split_status

    def reset(self):
        self._obj_log = {}
        self._registered_log_attr_by_get_dict = {}

    def flush(self):
        Logger().flush_recorder(recorder=self)

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


def record_return_decorator(which_recorder: str = 'global'):
    def wrap(fn):
        @wraps(fn)
        def wrap_with_self(self, *args, **kwargs):
            obj = self
            if which_recorder == 'global':
                recorder = get_global_recorder()
            elif which_recorder == 'self':
                recorder = getattr(obj, 'recorder')
            else:
                raise ValueError('Not supported recorder indicator: {}, use {}'.format(which_recorder, 'gloabl, self'))
            if not hasattr(obj, 'get_status') or not callable(obj.get_status):
                raise ValueError('registered obj {} mush have callable method get_status()'.format(type(obj)))
            res = fn(self, *args, **kwargs)
            if res is not None:
                info = obj.get_status()
                if not isinstance(res, dict):
                    raise TypeError('returned value by {} must be a dict in order to be recorded'.format(fn.__name__))
                for key, val in res.items():
                    recorder.append_to_obj_log(obj=obj, attr_name=key, status_info=info,
                                               log_val=val)
            return res

        return wrap_with_self

    return wrap


_global_recorder = Recorder()


def get_global_recorder() -> Recorder:
    return globals()['_global_recorder']


def reset_global_recorder():
    globals()['_global_recorder'].reset()


def reset_logging():
    Logger().reset()
    ConsoleLogger().reset()
    reset_global_recorder()
