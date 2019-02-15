import numpy as np


class ParameterAdaptiveWrapper(object):
    def __init__(self, source_obj, getter, setter):
        self.source_obj = source_obj
        self.getter = getter
        self.setter = setter
