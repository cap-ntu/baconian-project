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
from mobrl.core.core import Env
import abc
from mobrl.agent.agent import Agent
from mobrl.common.util.logging import Recorder
from mobrl.core.status import StatusCollector, StatusWithSingleInfo


class Experiment(Basic):
    STATUS_LIST = ('NOT_INIT', 'INITED', 'RUNNING', 'FINISHED', 'CORRUPTED')
    INIT_STATUS = 'NOT_INIT'
    required_key_dict = DictConfig.load_json(file_path=GlobalConfig.DEFAULT_EXPERIMENT_REQUIRED_KEY_LIST)

    @init_func_arg_record_decorator()
    @typechecked
    def __init__(self,
                 name: str,
                 agent: Agent,
                 env: Env,
                 config_or_config_dict: (DictConfig, dict),
                 tuner: Tuner = None,
                 ):
        super().__init__(status=StatusWithSingleInfo(obj=self), name=name)
        self.config = construct_dict_config(config_or_config_dict, self)
        self.agent = agent
        self.env = env
        self.tuner = tuner
        self.recorder = Recorder(flush_by_split_status=False)
        self.status_collector = StatusCollector()

        self._logger_kwargs = dict(config_or_config_dict=dict(),
                                   log_path=self.config('log_path'),
                                   log_level=self.config('log_level'))
        self._console_logger_kwargs = dict(
            to_file_flag=self.config('console_logger_to_file_flag'),
            to_file_name=self.config('console_logger_to_file_name'),
            level=self.config('log_level'),
            logger_name='console_logger'
        )
        self.status_collector.register_info_key_status(obj=agent, info_key='predict_counter',
                                                       under_status='TRAIN',
                                                       return_name='TOTAL_AGENT_TRAIN_SAMPLE_COUNT')
        self.status_collector.register_info_key_status(obj=agent, info_key='predict_counter',
                                                       under_status='TEST',
                                                       return_name='TOTAL_AGENT_TEST_SAMPLE_COUNT')
        self.status_collector.register_info_key_status(obj=agent,
                                                       info_key='update_counter',
                                                       under_status='TRAIN',
                                                       return_name='TOTAL_AGENT_UPDATE_COUNT')

    def init(self):
        self.agent.init()
        self.env.init()
        Logger().init(**self._logger_kwargs)
        ConsoleLogger().init(**self._console_logger_kwargs)
        self.set_status(val='INITED')

    def train(self):
        self.agent.train()

    def test(self):
        self.agent.test()

    def run(self):
        self.init()
        self.set_status('RUNNING')
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

    def _is_ended(self):
        res = self.status_collector()
        for key in GlobalConfig.DEFAULT_EXPERIMENT_END_POINT:
            if key in res and GlobalConfig.DEFAULT_EXPERIMENT_END_POINT[key] and res[key] >= \
                    GlobalConfig.DEFAULT_EXPERIMENT_END_POINT[key]:
                ConsoleLogger().print('info',
                                      'pipeline ended because {}: {} >= end point value {}'.format(key, res[key],
                                                                                                   GlobalConfig.DEFAULT_EXPERIMENT_END_POINT[
                                                                                                       key]))
                return True

        return False


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
