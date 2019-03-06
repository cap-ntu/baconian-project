import abc
from baconian.config.global_config import GlobalConfig
from baconian.common.logging import ConsoleLogger


class Flow(abc.ABC):
    @staticmethod
    def launch(func_dict: dict) -> bool:
        raise NotImplementedError


class TrainTestFlow(Flow):
    @staticmethod
    def launch(func_dict: dict) -> bool:
        try:
            while True:
                func_dict['train']['func'](*func_dict['train']['args'],
                                           **func_dict['train']['kwargs'])
                func_dict['test']['func'](*func_dict['test']['args'],
                                          **func_dict['test']['kwargs'])
                if func_dict['is_ended']['func'](*func_dict['is_ended']['args'],
                                                 **func_dict['is_ended']['kwargs']) is True:
                    break
            return True
        except GlobalConfig.DEFAULT_ALLOWED_EXCEPTION_OR_ERROR_LIST as e:
            ConsoleLogger().print('error', 'error {} occurred'.format(e))
            return False
