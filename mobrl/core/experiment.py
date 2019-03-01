"""
For experiments, it's functionality should include:
1. experiment and config set up
2. logging control
3. hyper-param tuning etc.
4. visualization
5. any related experiment utility
...
"""
from mobrl.core.core import Basic
from mobrl.core.status import StatusWithSingleInfo
from mobrl.core.pipeline import Pipeline
from mobrl.common.util.logging import Logger, ConsoleLogger
from mobrl.config.global_config import GlobalConfig
from mobrl.core.tuner import Tuner
from mobrl.config.dict_config import DictConfig
from mobrl.common.misc import *
from mobrl.core.util import init_func_arg_record_decorator
import tensorflow as tf
import numpy as np
import random
import GPUtil.GPUtil as Gpu
import os
from typeguard import typechecked
import time
from mobrl.tf.util import create_new_tf_session


class Experiment(Basic):
    STATUS_LIST = ('NOT_INIT', 'INITED', 'RUNNING', 'FINISHED', 'CORRUPTED')
    INIT_STATUS = 'NOT_INIT'
    required_key_dict = DictConfig.load_json(file_path=GlobalConfig.DEFAULT_EXPERIMENT_REQUIRED_KEY_LIST)

    @init_func_arg_record_decorator()
    @typechecked
    def __init__(self,
                 name: str,
                 pipeline: Pipeline,
                 config_or_config_dict: (DictConfig, dict),
                 tuner: Tuner = None,
                 ):
        super().__init__(status=StatusWithSingleInfo(obj=self), name=name)
        self.config = construct_dict_config(config_or_config_dict, self)
        self.pipeline = pipeline
        self.tuner = tuner
        self._logger_kwargs = dict(config_or_config_dict=dict(),
                                   log_path=self.config('log_path'),
                                   log_level=self.config('log_level'))
        self._console_logger_kwargs = dict(
            to_file_flag=self.config('console_logger_to_file_flag'),
            to_file_name=self.config('console_logger_to_file_name'),
            level=self.config('log_level'),
            logger_name='console_logger'
        )

    def init(self):
        Logger().init(**self._logger_kwargs)
        ConsoleLogger().init(**self._console_logger_kwargs)
        create_new_tf_session(cuda_device=0)
        self.set_status(val='INITED')

    @property
    def name(self):
        return self._name

    def run(self):
        self.init()
        self.set_status('RUNNING')
        self.pipeline.launch()
        self.set_status('FINISHED')
        self._exit()

    def _reset(self):
        pass

    def _exit(self):
        ConsoleLogger().flush()
        Logger().flush_recorder()
        ConsoleLogger().close()
        Logger().close()

        sess = tf.get_default_session()
        if sess:
            sess.__exit__(None, None, None)
        tf.reset_default_graph()


def _reset_global_seed(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@typechecked
def exp_runner(task_fn, auto_choose_gpu_flag=True, gpu_id: int = None, seed=None, **task_fn_kwargs):
    if auto_choose_gpu_flag is True:
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        DEVICE_ID_LIST = Gpu.getFirstAvailable()
        DEVICE_ID = DEVICE_ID_LIST[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if not seed:
        seed = int(round(time.time() * 1000)) % (2 ** 32 - 1)
    _reset_global_seed(seed)
    task_fn(**task_fn_kwargs)
