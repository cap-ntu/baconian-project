from typeguard import typechecked
from mobrl.config.dict_config import DictConfig
import abc
from mobrl.common.util.logging import Logger
import mobrl.common.util.files as files
import os


class Parameters(object):
    """
    A class that handle all parameters of a certain rl, to be better support in the future version.
    Currently, just a very simple implementation
    """

    @typechecked
    def __init__(self, parameters: dict, source_config: DictConfig = None, auto_init=False, name='parameters',
                 default_save_param_key=None):
        self._parameters = parameters
        self.name = name
        self._source_config = source_config if source_config else DictConfig(required_key_dict=dict(),
                                                                             config_dict=dict())
        self.default_save_param_key = default_save_param_key

        if auto_init is True:
            self.init()

    def __call__(self, key=None):
        if key:
            if isinstance(self._parameters, dict):
                if key in self._parameters:
                    return self._parameters[key]
                else:
                    return self._source_config(key)
            else:
                raise ValueError('parameters is not dict')
        else:
            # return self._parameters
            raise KeyError('specific a key to call {}'.format(type(self).__name__))

    @abc.abstractmethod
    def init(self):
        pass

    @abc.abstractmethod
    def copy_from(self, source_parameter):
        # todo implement this
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        # todo implement this
        raise NotImplementedError

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
        # for key, val in res.items():
        #     setattr(self, key, val)
