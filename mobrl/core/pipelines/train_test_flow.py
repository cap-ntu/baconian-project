import abc


class Flow(abc.ABC):
    @staticmethod
    def launch(func_dict: dict) -> bool:
        raise NotImplementedError


class TrainTestFlow(object):
    @staticmethod
    def launch(func_dict: dict) -> bool:
        while func_dict['is_ended'](*func_dict['is_ended']['args'],
                                    **func_dict['is_ended']['kwargs']) is False:
            func_dict['train']['func'](*func_dict['train']['args'],
                                       **func_dict['train']['kwargs'])


