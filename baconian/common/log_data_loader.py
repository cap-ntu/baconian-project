from baconian.common.plotter import Plotter
import glob
import os
import sys
from baconian.common.error import *
import json_tricks as json
import pandas as pd
from baconian.common.files import *
from pathlib import Path


class SingleExpLogDataLoader(object):
    def __init__(self, exp_root_dir: str):
        self._root_dir = exp_root_dir
        check_file(path=os.path.join(exp_root_dir, 'record', 'final_status.json'))  # dir list
        check_file(path=os.path.join(exp_root_dir, 'record', 'global_config.json'))
        self.final_status = load_json(file_path=os.path.join(exp_root_dir, 'record', 'final_status.json'))
        self.global_config = load_json(file_path=os.path.join(exp_root_dir, 'record', 'global_config.json'))
        # check the existence of json files?

    def load_record_data(self, agent_log_dir_name, algo_log_dir_name, env_log_dir_name):
        # todo maybe add a verbose mode to load all log

        agent_log_dir = os.path.join(self._root_dir, agent_log_dir_name)
        algo_log_dir = os.path.join(self._root_dir, algo_log_dir_name)
        check_dir(agent_log_dir)
        check_dir(algo_log_dir)

    def init(self):
        pass

    def plot_res(self, sub_log_dir_name, key, index, mode=('plot', 'hist', 'scatter'), average_over=1,
                 save_format=None):
        file_name = os.path.join(self._root_dir, 'record', sub_log_dir_name, 'TRAIN', 'log.json')
        f = open(file_name, 'r')
        res_dict = json.load(f)
        key_list = res_dict[key]
        key_value = {}
        key_vector = []
        index_vector = []
        for record in key_list:
            num_index = int(record[index])
            index_vector.append(num_index)
            key_vector.append(record["log_val"])
        key_value[index] = index_vector
        key_value[key] = key_vector
        data = pd.DataFrame.from_dict(key_value)  # Create dataframe for plotting
        row_num = data.shape[0]
        column_num = data.shape[1]

        # Calculate mean value in horizontal axis, incompatible with histogram mode
        if average_over != 1:
            new_row_num = int(row_num / average_over)
            data_new = data.head(new_row_num).copy()
            data_new.loc[:, index] = data_new.loc[:, index] * average_over
            for column in range(1, column_num):
                for i in range(new_row_num):
                    data_new.iloc[i, column] = data.iloc[i * average_over: i * average_over + average_over,
                                               column].mean()

        if mode == 'histogram':
            histogram_flag = True
            data_new = data.iloc[:, 1:].copy()
        else:
            histogram_flag = False
            data = data_new
        if mode == 'scatter':
            scatter_flag = True
        else:
            scatter_flag = False

        data = data_new

        Plotter.plot_any_key_in_log(data=data_new, index=index, key=key,
                                    sub_log_dir_name=sub_log_dir_name,
                                    scatter_flag=scatter_flag, save_flag=True,
                                    histogram_flag=histogram_flag, save_path=os.path.join(self._root_dir),
                                    save_format=save_format)


# TODO
# key and index, log given = plot figure(curve), average or 10, or set a range
# normalisation

class MultipleExpLogDataLoader(object):
    def __init__(self, exp_root_dir_list: str, num: int):
        self._root_dir = exp_root_dir_list
        self.num = num
        for i in range(num):
            exp_root_dir = exp_root_dir_list + "/exp_" + str(i)
            SingleExpLogDataLoader(exp_root_dir)

    def plot_res(self, key, index, sub_log_dir_name: str, mode=('plot', 'hist', 'scatter'), average_over=1,
                 save_format=None):
        multiple_key_value = {}
        for i in range(self.num):
            file_name = os.path.join(self._root_dir, 'exp_' + str(i), 'record', sub_log_dir_name, 'TRAIN', 'log.json')
            f = open(file_name, 'r')
            res_dict = json.load(f)
            key_list = res_dict[key]
            key_vector = []
            index_vector = []
            for record in key_list:
                num_index = int(record[index])
                index_vector.append(num_index)
                key_vector.append(record["log_val"])
            multiple_key_value[index] = index_vector
            multiple_key_value[key + '_' + str(i)] = key_vector

        data = pd.DataFrame.from_dict(multiple_key_value)  # Create dataframe for plotting
        row_num = data.shape[0]
        column_num = data.shape[1]

        # Calculate mean value in horizontal axis, incompatible with histogram mode
        if average_over != 1:
            new_row_num = int(row_num / average_over)
            data_new = data.head(new_row_num).copy()
            data_new.loc[:, index] = data_new.loc[:, index] * average_over
            for column in range(1, column_num):
                for i in range(new_row_num):
                    data_new.iloc[i, column] = data.iloc[i * average_over: i * average_over + average_over,
                                               column].mean()

        data['MEAN'] = data[data.columns[1:]].mean(axis=1)  # axis = 1 in columns, first column not counted
        data['STD_DEV'] = data[data.columns[1:-1]].std(axis=1)

        if mode == 'histogram':
            histogram_flag = True
            data_new = data.iloc[:, 1:-2].copy().stack()  # Mean and variance columns not counted
        else:
            histogram_flag = False
        if mode == 'scatter':
            scatter_flag = True
        else:
            scatter_flag = False

        Plotter.plot_any_key_in_log(data=data_new, index=index, key=key, exp_num=self.num,
                                    scatter_flag=scatter_flag, save_flag=True,
                                    mean_stddev_flag=True,
                                    histogram_flag=histogram_flag, save_path=os.path.join(self._root_dir),
                                    sub_log_dir_name=sub_log_dir_name, save_format=save_format)
