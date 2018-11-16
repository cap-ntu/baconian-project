# Date: 11/16/18
# Author: Luke
# Project: ModelBasedRLFramework
from src.core.basic import Basic
import typeguard as tg


class ValueFunction(Basic):

    @tg.typechecked
    def __init__(self, auto_set_up: bool = False):
        super().__init__()
        self._name = 'value_function'
        if auto_set_up:
            self.init_set_up()

    @tg.typechecked
    def copy(self, obj) -> bool:
        if isinstance(o=obj, t=type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        return True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    @tg.typechecked
    def init_set_up(self, *args, **kwargs) -> bool:
        return True


class TrainableValueFunction(ValueFunction):

    def __init__(self, auto_set_up: bool = False):
        super().__init__(auto_set_up)
        self.trainable_var_list = []
