import numpy as np
from typeguard import typechecked


class Parameters(object):
    """
    A class that handle all parameters of a certain rl, to be better support in the future version.
    Currently, just a very simple implementation
    """

    @typechecked
    def __init__(self, parameters: (list, dict, np.ndarray), auto_init=False, name='parameters'):
        self.parameters = parameters
        self.name = name
        if auto_init is True:
            self.init()

    def __call__(self, *args, **kwargs):
        return self.parameters

    def init(self):
        raise NotImplementedError

    def copy_from(self, source_parameter):
        raise NotImplementedError
