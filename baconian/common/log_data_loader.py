from baconian.common.plotter import Plotter
import glob
import os
import sys
from baconian.common.error import *
import json_tricks as json
from baconian.common.files import *
from pathlib import Path


class SingleExpLogDataLoader(object):
    def __init__(self, exp_root_dir: str):
        self._root_dir = exp_root_dir
        check_file(path=os.path.join(exp_root_dir, 'final_status.json'))
        check_file(path=os.path.join(exp_root_dir, 'global_config.json'))
        self.final_status = load_json(file_path=os.path.join(exp_root_dir, 'final_status.json'))
        self.global_config = load_json(file_path=os.path.join(exp_root_dir, 'global_config.json'))
        # check the existence of json files?

    def load_record_data(self, agent_log_dir_name, algo_log_dir_name, env_log_dir_name):
        # todo maybe add a verbose mode to load all log

        agent_log_dir = os.path.join(self._root_dir, 'record', agent_log_dir_name)
        algo_log_dir = os.path.join(self._root_dir, 'record', algo_log_dir_name)
        check_dir(agent_log_dir)
        check_dir(algo_log_dir)

    def init(self):
        pass

    def plot_res(self, obj_name, key, index, mode=('plot', 'hist', 'scatter'), ):
        file_name = obj_name
        f = open(file_name, 'r')
        #res = json.load(f) # res: dict
        #key_list = res[key]
        #for i in key_list:
         #   num_index = i[index]
          #  key[num_index] = i['log_val']
        if mode == 'hist':
            histogram_flag = True
        else:
            histogram_flag = False
        if mode == 'scatter':
            scatter_flag = True
        else:
            scatter_flag = False
        save_flag = True
        Plotter.plot_any_key_in_log(file_name, key, index, scatter_flag, save_flag,
                                    histogram_flag, )

# key and index, log given = plot figure(curve), average or 10, or set a range
# histogram
# mean & variance
# normalisation