# import numpy as np
# import glob
# import os
# import sys
# from baconian.common.error import *
# import json_tricks as json
# from baconian.common.files import *
#
#
# class SingleExpLogDataLoader(object):
#     def __init__(self, exp_root_dir: str):
#         self._root_dir = exp_root_dir
#         check_file(path=os.path.join(exp_root_dir, 'final_status.json'))
#         check_file(path=os.path.join(exp_root_dir, 'global_config.json'))
#         self.final_status = load_json(file_path=os.path.join(exp_root_dir, 'final_status.json'))
#         self.global_config = load_json(file_path=os.path.join(exp_root_dir, 'global_config.json'))
#
#     def load_record_data(self, agent_log_dir_name, algo_log_dir_name, env_log_dir_name):
#         # todo maybe add a verbose mode to load all log
#
#         agent_log_dir = os.path.join(self._root_dir, 'record', agent_log_dir_name)
#         algo_log_dir = os.path.join(self._root_dir, 'record', algo_log_dir_name)
#         check_dir(agent_log_dir)
#         check_dir(algo_log_dir)
#
#     def init(self):
#         pass
#
#     def plot_res(self, obj_name, key, index, mode=('plot', 'hist', 'scatter'), ):
#         pass
