from typeguard import typechecked
from mobrl.config.dict_config import DictConfig
import abc


class Parameters(object):
    """
    A class that handle all parameters of a certain rl, to be better support in the future version.
    Currently, just a very simple implementation
    """

    @typechecked
    def __init__(self, parameters: dict, source_config=None, auto_init=False, name='parameters'):
        self._parameters = parameters
        self.name = name
        self._source_config = source_config if source_config else DictConfig(required_key_dict=dict(),
                                                                             config_dict=dict())
        assert isinstance(self._source_config, DictConfig)

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
