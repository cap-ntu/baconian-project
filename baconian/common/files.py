import json_tricks as json
import os
import numpy as np
import shutil
import pandas as pd


def create_path(path, del_if_existed=True):
    if os.path.exists(path) is True and del_if_existed is False:
        raise FileExistsError()
    else:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
    os.makedirs(path)


def load_json(file_path) -> (dict, list):
    with open(file_path, 'r') as f:
        res = json.load(f)
        return res


def save_to_json(dict, path=None, fp=None, file_name=None):
    if fp:
        json.dump(obj=dict, fp=fp, indent=4, sort_keys=True)
        fp.close()
    else:
        if file_name is not None:
            path = os.path.join(path, file_name)
        with open(path, 'w') as f:
            json.dump(obj=dict, fp=f, indent=4, sort_keys=True)


def convert_dict_to_csv(log_dict):
    """
    This function will convert a log dict into csv file by recursively find the list in the
    dict and save the list with its key as a single csv file, the
    :param log_dict:
    :return: list, as the dict as each element with
    {'csv_file_name', 'csv_keys', 'csv_row_data'}
    """
    for val in log_dict:
        pass

# def numpy_to_json_serializable(source_log_content):
#     if isinstance(source_log_content, dict):
#         res = {}
#         for key, val in source_log_content.items():
#             if not isinstance(key, str):
#                 raise NotImplementedError('Not support the key of non-str type')
#             res[key] = numpy_to_json_serializable(val)
#         return res
#     elif isinstance(source_log_content, (list, tuple)):
#         res = []
#         for val in source_log_content:
#             res.append(numpy_to_json_serializable(val))
#         return res
#
#     elif isinstance(source_log_content, np.generic):
#         return source_log_content.item()
#     elif isinstance(source_log_content, np.generic):
#         return source_log_content.item()
#     else:
#         return source_log_content
