from typeguard import typechecked
from baconian.config.dict_config import DictConfig
import abc
from baconian.common.logging import Logger
import baconian.common.files as files
import os
from baconian.common.schedules import Scheduler
from copy import deepcopy, copy


class Parameter(object):
    # TODO
    def __init__(self):
        pass


class Parameters(object):
    """
    A class that handle all parameters of a certain rl, to be better support in the future version.
    Currently, just a very simple implementation
    """

    @typechecked
    def __init__(self, parameters: dict,
                 source_config: DictConfig = None,
                 name='parameters',
                 to_scheduler_param_tuple: tuple = None,
                 default_save_param_key=None):
        self._parameters = parameters
        self.name = name
        self._source_config = source_config if source_config else DictConfig(required_key_dict=dict(),
                                                                             config_dict=dict())
        self.default_save_param_key = default_save_param_key
        self._scheduler_info_dict = dict()
        self.to_scheduler_param_list = to_scheduler_param_tuple

    def __call__(self, key=None):
        if key:
            if key in self._scheduler_info_dict:
                new_val = self._scheduler_info_dict[key]['scheduler'].value()
                if key in self._parameters:
                    self._parameters[key] = new_val
                else:
                    self._source_config.set(key, new_val)
                return new_val
            if isinstance(self._parameters, dict):
                if key in self._parameters:
                    return self._parameters[key]
                else:
                    return self._source_config(key)
            else:
                raise ValueError('parameters is not dict')
        else:
            raise KeyError('specific a key to call {}'.format(type(self).__name__))

    def __getitem__(self, item):
        return self.__call__(key=item)

    def init(self):
        if self.to_scheduler_param_list:
            for val_dict in self.to_scheduler_param_list:
                self.set_scheduler(**val_dict)

    def copy_from(self, source_parameter):
        if not isinstance(source_parameter, type(self)):
            raise TypeError()

        self._update_dict(source_dict=source_parameter._parameters,
                          target_dict=self._parameters)
        self._source_config.config_dict = source_parameter._source_config.config_dict
        self.default_save_param_key = copy(source_parameter.default_save_param_key)
        if source_parameter.to_scheduler_param_list:
            self._scheduler_info_dict = dict()
            self.to_scheduler_param_list = copy(source_parameter.to_scheduler_param_list)
            if self.to_scheduler_param_list:
                for val_dict in self.to_scheduler_param_list:
                    self.set_scheduler(**val_dict)

    def _update_dict(self, source_dict: dict, target_dict: dict):
        for key, val in source_dict.items():
            target_dict[key] = val

    def save(self, save_path, global_step, name=None, default_save_param=None, *args, **kwargs):
        if default_save_param is None:
            default_save_param = dict(_parameters=self._parameters, _source_config=self._source_config.config_dict)
        if not name:
            name = self.name
        Logger().out_to_file(file_path=save_path,
                             file_name='{}-{}.json'.format(name, global_step),
                             content=default_save_param)

    def load(self, load_path, name, global_step, *args, **kwargs):
        res = files.load_json(file_path=os.path.join(load_path, "{}-{}.json".format(name, global_step)))
        # todo this mapping can be done via a dict structure
        if '_parameters' in res:
            setattr(self, '_parameters', res['_parameters'])
        if '_source_config' in res:
            setattr(self._source_config, 'config_dict', res['_source_config'])

    @typechecked
    def set_scheduler(self, param_key: str, scheduler: Scheduler, **kwargs):
        ori_value = self(param_key)
        scheduler.initial_p = ori_value
        self._scheduler_info_dict[param_key] = dict(param_key=param_key, scheduler=scheduler)

    def update(self, *args, **kwargs):
        for key, val in self._scheduler_info_dict.items():
            self.set(key=val['param_key'],
                     new_val=val['scheduler'].value())

    def set(self, key, new_val):
        if not isinstance(new_val, type(self(key))):
            raise TypeError('new value of parameters {} should be type {} instead of {}'.format(key, type(self(key)),
                                                                                                type(new_val)))
        elif key in self._parameters:
            self._parameters[key] = new_val
        else:
            self._source_config.set(key, new_val)
