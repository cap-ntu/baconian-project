import json_tricks as json
import os
import shutil
from baconian.common.error import *


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


def check_dir(path):
    if os.path.isdir(path) is False:
        raise LogPathOrFileNotExistedError('{} not existed'.format(path))


def check_file(path):
    if os.path.isfile(path) is False:
        raise LogPathOrFileNotExistedError('{} not existed'.format(path))


def save_to_json(obj: (list, dict), path=None, fp=None, file_name=None):
    jsonable_dict = convert_to_jsonable(dict_or_list=obj)
    if fp:
        json.dump(obj=jsonable_dict, fp=fp, indent=4, sort_keys=True)
        fp.close()
    else:
        if file_name is not None:
            path = os.path.join(path, file_name)
        with open(path, 'w') as f:
            json.dump(obj=obj, fp=f, indent=4, sort_keys=True)


def convert_to_jsonable(dict_or_list) -> (list, dict):
    if isinstance(dict_or_list, list):
        jsonable_dict = []
        for val in dict_or_list:
            if isinstance(val, (dict, list)):
                res = convert_to_jsonable(dict_or_list=val)
                jsonable_dict.append(res)
            else:
                f = open(os.devnull, 'w')
                try:
                    json.dump([val], f)
                except Exception:
                    jsonable_dict.append(str(val))
                else:
                    jsonable_dict.append(val)
                finally:
                    f.close()
        return jsonable_dict

    elif isinstance(dict_or_list, dict):
        jsonable_dict = dict()
        for key, val in dict_or_list.items():
            if isinstance(val, (dict, list)):
                res = convert_to_jsonable(dict_or_list=val)
                jsonable_dict[key] = res
            else:
                f = open(os.devnull, 'w')
                try:
                    json.dump([val], f)
                except Exception:
                    jsonable_dict[key] = str(val)
                else:
                    jsonable_dict[key] = val
                finally:
                    f.close()
        return jsonable_dict


def convert_dict_to_csv(log_dict):
    """
    This function will convert a log dict into csv file by recursively find the list in the
    dict and save the list with its key as a single csv file, the
    :param log_dict:
    :return: list, as the dict as each element with
    {'csv_file_name', 'csv_keys', 'csv_row_data'}
    """
    raise NotImplementedError
