import os
import random
import time

import numpy as np
import tensorflow as tf
from GPUtil import GPUtil as Gpu
from typeguard import typechecked

from baconian.common.util import files as file
from baconian.common.util.logging import Logger, ConsoleLogger
from baconian.config.global_config import GlobalConfig
from copy import deepcopy


def _reset_global_seed(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@typechecked
def single_exp_runner(task_fn, auto_choose_gpu_flag=True, gpu_id: int = None, seed=None, **task_fn_kwargs):
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
    file.create_path(path=GlobalConfig.DEFAULT_LOG_PATH, del_if_existed=True)
    Logger().init(config_or_config_dict=dict(),
                  log_path=GlobalConfig.DEFAULT_LOG_PATH,
                  log_level=GlobalConfig.DEFAULT_LOG_LEVEL)
    ConsoleLogger().init(to_file_flag=GlobalConfig.DEFAULT_WRITE_CONSOLE_LOG_TO_FILE_FLAG,
                         to_file_name=os.path.join(GlobalConfig.DEFAULT_LOG_PATH,
                                                   GlobalConfig.DEFAULT_CONSOLE_LOG_FILE_NAME),
                         level=GlobalConfig.DEFAULT_LOG_LEVEL,
                         logger_name=GlobalConfig.DEFAULT_CONSOLE_LOGGER_NAME)

    task_fn(**task_fn_kwargs)


@typechecked
def duplicate_exp_runner(num, task_fn, auto_choose_gpu_flag=True, gpu_id: int = None, seeds: list = None,
                         **task_fn_kwargs):
    if seeds:
        assert len(seeds) == num
    base_log_path = deepcopy(GlobalConfig.DEFAULT_LOG_PATH)

    for i in range(num):
        GlobalConfig.set('DEFAULT_LOG_PATH', os.path.join(base_log_path, 'exp_{}'.format(i)))
        single_exp_runner(task_fn=task_fn, auto_choose_gpu_flag=auto_choose_gpu_flag,
                          gpu_id=gpu_id, seed=seeds[i] if seeds else None, **task_fn_kwargs)
