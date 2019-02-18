"""
For experiments, it's functionality should include:
1. experiment and config set up
2. logging control
3. hyper-param tuning etc.
4. visualization
5. any related experiment utility
...
"""
from mbrl.core.basic import Basic
from mbrl.core.status import StatusWithSingleInfo
from mbrl.core.pipeline import Pipeline
from mbrl.common.util.logger import Logger
from mbrl.core.tuner import Tuner
from typeguard import typechecked
from mbrl.config.dict_config import DictConfig
from mbrl.common.misc import *
from mbrl.core.util import init_func_arg_record_decorator


class Experiment(Basic):
    STATUS_LIST = ('NOT_INIT', 'INITED', 'RUNNING', 'FINISHED', 'CORRUPTED')
    INIT_STATUS = 'NOT_INIT'

    @init_func_arg_record_decorator()
    @typechecked
    def __init__(self,
                 pipeline: Pipeline,
                 logger: Logger,
                 tuner: Tuner,
                 config_or_config_dict: (DictConfig, dict),
                 ):
        super().__init__(StatusWithSingleInfo(obj=self))
        self._name = 'experiment'
        self.config = construct_dict_config(config_or_config_dict, self)
        self.pipeline = pipeline
        self.logger = logger
        self.tuner = tuner

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
