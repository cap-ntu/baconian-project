import abc
from baconian.config.global_config import GlobalConfig
from baconian.common.logging import ConsoleLogger
from baconian.config.dict_config import DictConfig
from baconian.common.misc import *
from baconian.core.parameters import Parameters
from baconian.core.status import *


class Flow(abc.ABC):
    required_func = ()
    required_key_dict = dict()

    def __init__(self, func_dict):
        self.func_dict = func_dict
        for key in self.required_func:
            assert key in func_dict

    def launch(self) -> bool:
        try:
            return self._launch()
        except GlobalConfig().DEFAULT_ALLOWED_EXCEPTION_OR_ERROR_LIST as e:
            ConsoleLogger().print('error', 'error {} occurred'.format(e))
            return False

    def _launch(self) -> bool:
        raise NotImplementedError

    def _call_func(self, key, **extra_kwargs):
        return self.func_dict[key]['func'](*self.func_dict[key]['args'],
                                           **extra_kwargs,
                                           **self.func_dict[key]['kwargs'])


class TrainTestFlow(Flow):
    required_func = ('train', 'test', 'sample')
    required_key_dict = {
        "TEST_EVERY_SAMPLE_COUNT": 1000,
        "TRAIN_EVERY_SAMPLE_COUNT": 1000,
        "START_TRAIN_AFTER_SAMPLE_COUNT": 1,
        "START_TEST_AFTER_SAMPLE_COUNT": 1,
    }

    def __init__(self,
                 train_sample_count_func,
                 config_or_config_dict: (DictConfig, dict),
                 func_dict: dict,
                 ):
        super(TrainTestFlow, self).__init__(func_dict=func_dict)
        config = construct_dict_config(config_or_config_dict, obj=self)
        self.parameters = Parameters(source_config=config, parameters=dict())
        self.time_step_func = train_sample_count_func
        self.last_train_point = -1
        self.last_test_point = -1
        assert callable(train_sample_count_func)

    def _launch(self) -> bool:
        while True:
            self._call_func('sample')
            if self.time_step_func() - self.parameters(
                    'TRAIN_EVERY_SAMPLE_COUNT') >= self.last_train_point and self.time_step_func() > self.parameters(
                'START_TRAIN_AFTER_SAMPLE_COUNT'):
                self.last_train_point = self.time_step_func()
                self._call_func('train')
            if self.time_step_func() - self.parameters(
                    'TEST_EVERY_SAMPLE_COUNT') >= self.last_test_point and self.time_step_func() > self.parameters(
                'START_TEST_AFTER_SAMPLE_COUNT'):
                self.last_test_point = self.time_step_func()
                self._call_func('test')
            if self._is_ended() is True:
                break
        return True

    def _is_ended(self):
        key_founded_flag = False
        finished_flag = False
        for key in GlobalConfig().DEFAULT_EXPERIMENT_END_POINT:
            if GlobalConfig().DEFAULT_EXPERIMENT_END_POINT[key] is not None:
                key_founded_flag = True
                if get_global_status_collect()(key) >= GlobalConfig().DEFAULT_EXPERIMENT_END_POINT[key]:
                    ConsoleLogger().print('info',
                                          'pipeline ended because {}: {} >= end point value {}'.
                                          format(key, get_global_status_collect()(key),
                                                 GlobalConfig().DEFAULT_EXPERIMENT_END_POINT[key]))
                    finished_flag = True
        if key_founded_flag is False:
            ConsoleLogger().print(
                'warning',
                '{} in experiment_end_point is not registered with global status collector: {}, experiment may not end'.
                    format(GlobalConfig().DEFAULT_EXPERIMENT_END_POINT, list(get_global_status_collect()().keys())))
        return finished_flag
