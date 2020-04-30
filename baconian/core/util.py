from copy import deepcopy
from collections import Hashable
from baconian.common.error import *
from baconian.core.global_var import get_all, reset
from functools import wraps
from baconian.config.global_config import GlobalConfig


def init_func_arg_record_decorator():
    def wrap(fn):
        @wraps(fn)
        def wrap_with_self(self, *args, **kwargs):
            get_all()['_global_obj_arg_dict'][self] = None
            # get_all()['_global_obj_arg_dict'][self] = dict(args=args, kwargs=kwargs, cls=type(self))
            res = fn(self, *args, **kwargs)
            return res

        return wrap_with_self

    return wrap


def get_global_arg_dict():
    return get_all()['_global_obj_arg_dict']


def copy_globally(arg_dict, source_obj_list):
    """
    deprecated in the future
    :param arg_dict:
    :param source_obj_list:
    :return:
    """
    new_obj_list = []
    reset('_global_name_dict')
    for obj in source_obj_list:
        if obj not in arg_dict:
            raise ValueError('{} not in arg_dict'.format(obj))
        else:
            new_obj_list.append(_make_copy_object(arg_dict, obj=obj))

    return new_obj_list


def register_name_globally(name: str, obj):
    if name in get_all()['_global_name_dict'] and not id(obj) == id(
            get_all()['_global_name_dict'][name]) and obj.allow_duplicate_name is False and \
            get_all()['_global_name_dict'][
                name].allow_duplicate_name is False and GlobalConfig().DEFAULT_TURN_OFF_GLOBAL_NAME_FLAG is False:
        raise GlobalNameExistedError(
            'name : {} is existed with object: {}'.format(name, get_all()['_global_name_dict'][name]))
    else:
        get_all()['_global_name_dict'][name] = obj


def _make_copy_object(arg_dict: dict, obj):
    if obj not in arg_dict:
        raise ValueError('{} not in arg_dict'.format(obj))
    else:
        args = arg_dict[obj]['args']
        kwargs = arg_dict[obj]['kwargs']
        cls = arg_dict[obj]['cls']

        new_args = []
        new_kwargs = dict()

        arg_dict.pop(obj)
        del obj
        for a in args:
            if in_dict(a, args):
                new_args.append(_make_copy_object(arg_dict, obj=a))
            else:
                new_args.append(deepcopy(a))
        for key, a in kwargs.items():
            print(key, a)
            if in_dict(a, arg_dict):
                new_kwargs[key] = _make_copy_object(arg_dict, obj=a)
            else:
                new_kwargs[key] = deepcopy(a)
        print("create ", cls, new_args, new_kwargs, flush=True)
        return cls(*new_args, **new_kwargs)


def in_dict(obj, list_or_dict):
    if isinstance(obj, Hashable):
        return obj in list_or_dict
    else:
        if isinstance(list_or_dict, (tuple, list)):
            for key in list_or_dict:
                if id(key) == id(obj):
                    return True
            return False
        elif isinstance(list_or_dict, dict):
            for key, val in list_or_dict.items():
                if id(key) == id(obj):
                    return True
            return False
