import numpy as np
from matplotlib import pyplot as plt
import json
import seaborn as sns
import os
import glob
from baconian.common.log_data_loader import *

exp_root_dir_list = ['/home/cly/Documents/baconian-internal/benchmark/benchmark_log/CartPole-v1/dqn/2019-11-14_16-30-48',
                     '/home/cly/Documents/baconian-internal/benchmark/benchmark_log/CartPole-v1/dqn/2019-11-11_13-10-10',
                    ]

if __name__ == "__main__":
    MultipleExpLogDataLoader(exp_root_dir_list=exp_root_dir_list)\
                         .plot_res(sub_log_dir_name='benchmark_agent/TRAIN',
                         key='sum_reward', index='sample_counter',
                         mode='line', average_over=1, file_name=None, save_format='png', save_path='/home/cly/Document'
                                                                                                   's/baconian-internal/benchmark/')