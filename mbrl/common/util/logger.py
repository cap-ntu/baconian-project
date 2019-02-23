import os
import json
import numpy as np
from typeguard import typechecked
from mbrl.config.dict_config import DictConfig
from mbrl.common.misc import *
from mbrl.config.global_config import GlobalConfig
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
        self._registered_recorder = []
        self._log_dir = None
        self._config_file_log_dir = None
        self._loss_file_log_dir = None
        self._model_file_log_dir = None
        self.logger_config = None
        self.log_level = None
        self.inited_flag = False

    def init(self, config_or_config_dict: (DictConfig, dict),
             log_path, log_level=None, **kwargs):
        self._log_dir = log_path
        self._config_file_log_dir = None
        self._loss_file_log_dir = None
        self._model_file_log_dir = None
        # todo debug mode
        # if os.path.exists(self._log_dir):
        #     raise FileExistsError('%s path is existed' % self._log_dir)
        self.logger_config = construct_dict_config(config_or_config_dict, obj=self)
        self.log_level = log_level
        self.inited_flag = True

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

    def _save(self):
        # todo
        pass

    def flush(self):
        # todo
        pass

    def close(self):
        self._save()

    def append_recorder(self, recorder):
        self._registered_recorder.append(recorder)

    def reset(self):
        self._registered_recorder = []
        self.inited_flag = False


class Logger(object):
    only_instance = None

    def __new__(cls, *args, **kwargs):
        if Logger.only_instance is None:
            Logger.only_instance = _SingletonLogger()
        return Logger.only_instance

    def init(self, config_or_config_dict: (DictConfig, dict),
             log_path, log_level=None, **kwargs):
        if not Logger.only_instance.inited_flag:
            Logger.only_instance.init(config_or_config_dict=config_or_config_dict,
                                      log_path=log_path,
                                      log_level=log_level,
                                      **kwargs)

    def reset(self):
        Logger.only_instance.reset()


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
        self.name = logger_name
        logging.basicConfig(format=GlobalConfig.DEFAULT_LOGGING_FORMAT)
        if level not in self.ALLOWED_LOG_LEVEL:
            raise ValueError('Wrong log level use {} instead'.format(self.ALLOWED_LOG_LEVEL))
        self.logger = logging.getLogger('console_logger')
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

    def close(self):
        if self.file_handler:
            self.file_handler.close()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    def reset(self):
        self.close()
        self.inited_flag = False


class ConsoleLogger(object):
    only_instance = None

    def __new__(cls, *args, **kwargs):
        if not ConsoleLogger.only_instance:
            ConsoleLogger.only_instance = _SingletonConsoleLogger()
        return ConsoleLogger.only_instance

    def init(self, logger_name, to_file_flag, level: str, to_file_name: str = None):
        if not ConsoleLogger.only_instance.inited_flag:
            ConsoleLogger.only_instance.init(logger_name=logger_name, to_file_flag=to_file_flag,
                                             level=level, to_file_name=to_file_name)

    def reset(self):
        ConsoleLogger.only_instance.reset()


global_console_logger = ConsoleLogger()
