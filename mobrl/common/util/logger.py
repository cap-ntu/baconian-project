import os
import json
import numpy as np
from typeguard import typechecked
from mobrl.config.dict_config import DictConfig
from mobrl.common.misc import *
from mobrl.config.global_config import GlobalConfig
from copy import deepcopy
import logging
import abc

"""
Logger Module
1. a global console output file
2. each module/instance will have a single log file
3. support for tf related utility, tf model, tensorboard
4. support for different log file types
5. support for different level of logging
"""


class BaseLogger(object):
    def __init__(self):
        self.inited_flag = False

    @abc.abstractmethod
    def close(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def init(self, *args, **kwargs):
        raise NotImplementedError


class _SingletonLogger(BaseLogger):
    """
    A private class that should never be instanced, it is used to implement the singleton design pattern for Logger
    """

    required_key_list = []

    def __init__(self):
        super(_SingletonLogger, self).__init__()
        self._registered_recorders = []
        self._log_dir = None
        self._config_file_log_dir = None
        self._record_file_log_dir = None
        self.logger_config = None
        self.log_level = None
        self.inited_flag = False

    def init(self, config_or_config_dict: (DictConfig, dict),
             log_path, log_level=None, **kwargs):
        if self.inited_flag:
            return
        self._log_dir = log_path
        # todo debug only
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
        self.flush_recorder()

    def append_recorder(self, recorder):
        self._registered_recorders.append(recorder)

    def reset(self):
        self._registered_recorders = []
        self.inited_flag = False

    def _flush(self, recorder):
        if recorder.is_empty():
            return
        from mobrl.common.util.recorder import Recorder
        assert isinstance(recorder, Recorder)
        log_dict, by_status_flag = recorder.get_obj_log_to_flush(clear_obj_log_flag=True)
        for obj_name, obj_log_dict in log_dict.items():
            if by_status_flag is True:
                for status, status_log_dict in obj_log_dict.items():
                    self._out_to_file(
                        file_path=os.path.join(self._record_file_log_dir, str(obj_name), str(status)),
                        content=status_log_dict,
                        file_name='log.json')
            else:
                self._out_to_file(file_path=os.path.join(self._record_file_log_dir, str(obj_name)),
                                  content=obj_log_dict,
                                  file_name='log.json')

        # by_status_flag = true: [obj][status][attr_name]
        # by_status_flag = false: [obj][attr_name]

    @typechecked
    def _out_to_file(self, file_path: str, content: (tuple, list, dict), file_name: str):
        if len(content) == 0:
            return
        # ConsoleLogger().print('info', 'Write log to %s/%s', file_path, file_name)
        ConsoleLogger().print('info', 'Write log to {}/{}'.format(file_path, file_name))
        mode = 'a'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            mode = 'w'
        try:
            f = open(os.path.join(file_path, file_name), mode)
        except FileNotFoundError:
            f = open(file_path, 'w')

        content = self._numpy_to_json_serializable(source_log_content=content)
        json.dump(content, fp=f, indent=4, sort_keys=True)
        f.close()

    def _numpy_to_json_serializable(self, source_log_content):
        if isinstance(source_log_content, dict):
            res = {}
            for key, val in source_log_content.items():
                if not isinstance(key, str):
                    raise NotImplementedError('Not support the key of non-str type')
                res[key] = self._numpy_to_json_serializable(val)
            return res
        elif isinstance(source_log_content, (list, tuple)):
            res = []
            for val in source_log_content:
                res.append(self._numpy_to_json_serializable(val))
            return res

        elif isinstance(source_log_content, np.generic):
            return source_log_content.item()
        else:
            return source_log_content


class Logger(object):
    only_instance = None

    def __new__(cls, *args, **kwargs):
        if Logger.only_instance is None:
            Logger.only_instance = _SingletonLogger()
        return Logger.only_instance

    # def init(self, config_or_config_dict: (DictConfig, dict),
    #          log_path, log_level=None, **kwargs):
    #     if not Logger.only_instance.inited_flag:
    #         Logger.only_instance.init(config_or_config_dict=config_or_config_dict,
    #                                   log_path=log_path,
    #                                   log_level=log_level,
    #                                   **kwargs)
    #
    # def reset(self):
    #     Logger.only_instance.reset()


global_logger = Logger()


class _SingletonConsoleLogger(BaseLogger):
    """
    A private class that should never be instanced, it is used to implement the singleton design pattern for
    ConsoleLogger
    """
    ALLOWED_LOG_LEVEL = ('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET')
    ALLOWED_PRINT_TYPE = ('info', 'warning', 'debug', 'critical', 'log', 'critical')

    def __init__(self):
        super(_SingletonConsoleLogger, self).__init__()
        self.name = None
        self.logger = None
        self.file_handler = None
        self.inited_flag = False

    def init(self, logger_name, to_file_flag, level: str, to_file_name: str = None):
        if self.inited_flag is True:
            return
        self.name = logger_name
        logging.basicConfig(format=GlobalConfig.DEFAULT_LOGGING_FORMAT)
        if level not in self.ALLOWED_LOG_LEVEL:
            raise ValueError('Wrong log level use {} instead'.format(self.ALLOWED_LOG_LEVEL))
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, level))

        self.file_handler = None
        if to_file_flag is True:
            self.file_handler = logging.FileHandler(filename=to_file_name)
            self.file_handler.setFormatter(fmt=logging.Formatter(fmt=GlobalConfig.DEFAULT_LOGGING_FORMAT))
            self.file_handler.setLevel(getattr(logging, level))
            self.logger.addHandler(self.file_handler)
        self.inited_flag = True

    def print(self, p_type: str, p_str: str, *arg, **kwargs):
        if p_type not in self.ALLOWED_PRINT_TYPE:
            raise ValueError('use print type from {}'.format(self.ALLOWED_PRINT_TYPE))
        getattr(self.logger, p_type)(p_str, *arg, **kwargs)
        self.flush()

    def close(self):
        if self.file_handler:
            self.file_handler.close()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    def reset(self):
        self.close()
        self.inited_flag = False

    def flush(self):
        if self.file_handler:
            self.file_handler.flush()
        for handler in logging.root.handlers[:]:
            handler.flush()


class ConsoleLogger(object):
    only_instance = None

    def __new__(cls, *args, **kwargs):
        if not ConsoleLogger.only_instance:
            ConsoleLogger.only_instance = _SingletonConsoleLogger()
        return ConsoleLogger.only_instance

    # def init(self, logger_name, to_file_flag, level: str, to_file_name: str = None):
    #     if not ConsoleLogger.only_instance.inited_flag:
    #         ConsoleLogger.only_instance.init(logger_name=logger_name, to_file_flag=to_file_flag,
    #                                          level=level, to_file_name=to_file_name)
    #
    # def reset(self):
    #     ConsoleLogger.only_instance.reset()


global_console_logger = ConsoleLogger()
