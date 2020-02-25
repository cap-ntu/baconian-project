from baconian.common.plotter import Plotter
import pandas as pd
from baconian.common.files import *
from collections import OrderedDict
from typing import Union


class SingleExpLogDataLoader(object):
    def __init__(self, exp_root_dir: str):
        self._root_dir = exp_root_dir
        check_file(path=os.path.join(exp_root_dir, 'record', 'final_status.json'))  # dir list
        check_file(path=os.path.join(exp_root_dir, 'record', 'global_config.json'))
        self.final_status = load_json(file_path=os.path.join(exp_root_dir, 'record', 'final_status.json'))
        self.global_config = load_json(file_path=os.path.join(exp_root_dir, 'record', 'global_config.json'))

    def load_record_data(self, agent_log_dir_name, algo_log_dir_name, env_log_dir_name):
        # TODO  pre-load all data here
        check_dir(os.path.join(self._root_dir, agent_log_dir_name))
        check_dir(os.path.join(self._root_dir, algo_log_dir_name))

    def init(self):
        pass

    def plot_res(self, sub_log_dir_name, key, index, save_path=None, mode=('line', 'hist', 'scatter'),
                 average_over=1, file_name=None, save_format='png', save_flag=False,
                 ):
        log_name = os.path.join(self._root_dir, 'record', sub_log_dir_name, 'log.json')
        f = open(log_name, 'r')
        res_dict = json.load(f)
        key_list = res_dict[key]
        key_value = OrderedDict()
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
        data_new = data

        # Calculate mean value in horizontal axis, incompatible with histogram mode
        if average_over != 1:
            if mode != 'histogram':
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
        scatter_flag = True if mode == 'scatter' else False

        Plotter.plot_any_key_in_log(data=data_new, index=index, key=key,
                                    sub_log_dir_name=sub_log_dir_name,
                                    scatter_flag=scatter_flag, save_flag=save_flag,
                                    histogram_flag=histogram_flag, save_path=save_path,
                                    save_format=save_format, file_name=file_name)


# TODO
# key and index, log given = plot figure(curve), average or 10, or set a range
# normalisation

class MultipleExpLogDataLoader(object):
    def __init__(self, exp_root_dir_list: Union[str, list]):
        self._root_dir = exp_root_dir_list
        self.exp_list = []
        self.num = 0
        if type(self._root_dir) is str:
            for path in os.listdir(self._root_dir):
                print('path: ', path)
                exp_root_dir = os.path.join(self._root_dir, path)
                print(exp_root_dir)
                self.exp_list.append(exp_root_dir)
                self.num += 1
                SingleExpLogDataLoader(exp_root_dir)
        else:
            for dependent_exp in self._root_dir:
                assert type(dependent_exp) is str
                for path in os.listdir(dependent_exp):
                    exp_root_dir = os.path.join(dependent_exp, path)
                    self.exp_list.append(exp_root_dir)
                    self.num += 1
                    SingleExpLogDataLoader(exp_root_dir)

    def plot_res(self, key, index, save_path, sub_log_dir_name: str, mode=('plot', 'hist', 'scatter'), average_over=1,
                 save_format='png', file_name=None, save_flag=False):
        multiple_key_value = {}
        for exp in self.exp_list:
            f = open(os.path.join(exp, 'record', sub_log_dir_name, 'log.json'), 'r')
            res_dict = json.load(f)
            key_list = res_dict[key]
            key_vector = []
            index_vector = []
            for record in key_list:
                num_index = int(record[index])
                index_vector.append(num_index)
                key_vector.append(record["log_val"])
            multiple_key_value[index] = index_vector
            multiple_key_value[key + '_' + exp] = key_vector
        data = pd.DataFrame.from_dict(multiple_key_value)  # Create dataframe for plotting
        row_num = data.shape[0]
        column_num = data.shape[1]
        data_new = data

        # Calculate mean value in horizontal axis, incompatible with histogram mode
        if average_over != 1:
            if mode != 'histogram':
                new_row_num = int(row_num / average_over)
                data_new = data.head(new_row_num).copy()
                data_new.loc[:, index] = data_new.loc[:, index] * average_over
                for column in range(1, column_num):
                    for i in range(new_row_num):
                        data_new.iloc[i, column] = data.iloc[i * average_over: i * average_over + average_over,
                                                   column].mean()

        data_new['MEAN'] = data_new[data_new.columns[1:]].mean(axis=1)  # axis = 1 in columns, first column not counted
        data_new['STD_DEV'] = data_new[data_new.columns[1:-1]].std(axis=1)

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
                                    scatter_flag=scatter_flag, save_flag=save_flag,
                                    mean_stddev_flag=True,
                                    histogram_flag=histogram_flag, save_path=save_path,
                                    sub_log_dir_name=sub_log_dir_name, save_format=save_format, file_name=file_name)
