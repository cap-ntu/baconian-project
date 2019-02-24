"""
For experiments, it's functionality should include:
1. experiment and config set up
2. logging control
3. hyper-param tuning etc.
4. visualization
5. any related experiment utility
...
"""
from mobrl.core.basic import Basic
from mobrl.core.status import StatusWithSingleInfo
from mobrl.core.pipeline import Pipeline
from mobrl.common.util.logger import global_logger, global_console_logger
from mobrl.config.global_config import GlobalConfig
from mobrl.core.tuner import Tuner
from typeguard import typechecked
from mobrl.config.dict_config import DictConfig
from mobrl.common.misc import *
from mobrl.core.util import init_func_arg_record_decorator


class Experiment(Basic):
    STATUS_LIST = ('NOT_INIT', 'INITED', 'RUNNING', 'FINISHED', 'CORRUPTED')
    INIT_STATUS = 'NOT_INIT'

    @init_func_arg_record_decorator()
    @typechecked
    def __init__(self,
                 pipeline: Pipeline,
                 tuner: Tuner,
                 config_or_config_dict: (DictConfig, dict),
                 # todo simplify the parameters
                 log_path: str = GlobalConfig.DEFAULT_LOG_PATH,
                 log_level: str = GlobalConfig.DEFAULT_LOG_LEVEL,
                 log_use_global_memo=GlobalConfig.DEFAULT_LOG_USE_GLOBAL_MEMO_FLAG,
                 console_log_to_file_flag=GlobalConfig.DEFAULT_WRITE_CONSOLE_LOG_TO_FILE_FLAG,
                 console_log_to_file_name=GlobalConfig.DEFAULT_CONSOLE_LOG_FILE_NAME,
                 ):
        super().__init__(StatusWithSingleInfo(obj=self))
        self._name = 'experiment'
        self.config = construct_dict_config(config_or_config_dict, self)
        self.pipeline = pipeline
        self.tuner = tuner
        self._logger_kwargs = dict(config_or_config_dict=dict(),
                                   log_path=log_path,
                                   log_level=log_level,
                                   use_global_memo=log_use_global_memo)
        self._console_logger_kwargs = dict(
            to_file_flag=console_log_to_file_flag,
            to_file_name=console_log_to_file_name,
            level=log_level,
            logger_name='console_logger'
        )

    def init(self):
        global_logger.init(**self._logger_kwargs)
        global_console_logger.init(**self._console_logger_kwargs)

    @property
    def name(self):
        return self._name

    def run(self):
        self.pipeline.launch()

    def _reset(self):
        pass

    def _exit(self):
        pass


class ExperimentSetter(object):
    """
    recursively set up a experiment object
    """

    def set_experiment(self, source_exp: Experiment, exp_dict: dict):
        pass
