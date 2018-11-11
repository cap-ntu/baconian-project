from src.core.basic import Basic
import abc


class Algo(Basic, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def optimize(self, *arg, **kwargs):
        pass

    @abc.abstractmethod
    def output(self, *arg, **kwargs):
        pass
