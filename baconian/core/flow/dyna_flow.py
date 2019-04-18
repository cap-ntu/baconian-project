from baconian.core.flow.train_test_flow import Flow
from baconian.config.global_config import GlobalConfig
from baconian.common.logging import ConsoleLogger
from baconian.config.dict_config import DictConfig
from baconian.common.misc import *
from baconian.core.parameters import Parameters
from baconian.core.status import *


class DynaFlow(Flow):
    """
    A typical flow for utilizing the model-based algo, it is not restricted to Dyna algorithms,
    but can be utilized by others.
    """

    required_func = ('train_algo', 'train_algo_from_synthesized_data', 'train_dynamics', 'test_algo', 'test_dynamics',
                     'sample_from_real_env', 'sample_from_dynamics_env')

    required_key_dict = {
        "TEST_ALGO_EVERY_REAL_SAMPLE_COUNT": 1000,
        "TEST_DYNAMICS_EVERY_REAL_SAMPLE_COUNT": 1000,
        "TRAIN_ALGO_EVERY_REAL_SAMPLE_COUNT": 1000,
        "TRAIN_DYNAMICS_EVERY_REAL_SAMPLE_COUNT": 1000,
        "START_TRAIN_ALGO_AFTER_SAMPLE_COUNT": 1,
        "START_TRAIN_DYNAMICS_AFTER_SAMPLE_COUNT": 1,
        "START_TEST_ALGO_AFTER_SAMPLE_COUNT": 1,
        "START_TEST_DYNAMICS_AFTER_SAMPLE_COUNT": 1,
        "WARM_UP_DYNAMICS_SAMPLES": 1000
    }

    def __init__(self,
                 train_sample_count_func,
                 config_or_config_dict: (DictConfig, dict),
                 func_dict: dict, ):
        super().__init__(func_dict)
        super(DynaFlow, self).__init__(func_dict=func_dict)
        config = construct_dict_config(config_or_config_dict, obj=self)
        self.parameters = Parameters(source_config=config, parameters=dict())
        self.time_step_func = train_sample_count_func
        self._last_train_algo_point = -1
        self._last_test_algo_point = -1
        self._last_train_dynamics_point = -1
        self._last_test_dynamics_point = -1
        assert callable(train_sample_count_func)

    def _launch(self) -> bool:

        while True:
            real_batch_data = self._call_func('sample_from_real_env')
            if self.time_step_func() - self.parameters(
                    'TRAIN_ALGO_EVERY_REAL_SAMPLE_COUNT') >= self._last_train_algo_point and self.time_step_func() > self.parameters(
                'START_TRAIN_ALGO_AFTER_SAMPLE_COUNT'):
                self._last_train_algo_point = self.time_step_func()
                self._call_func('train_algo')

            if self.time_step_func() - self.parameters(
                    'TRAIN_DYNAMICS_EVERY_REAL_SAMPLE_COUNT') >= self._last_train_dynamics_point and self.time_step_func() > self.parameters(
                'START_TRAIN_DYNAMICS_AFTER_SAMPLE_COUNT'):
                self._last_train_algo_point = self.time_step_func()
                self._call_func('train_dynamics', batch_data=real_batch_data)
            if self.time_step_func() >= self.parameters('WARM_UP_DYNAMICS_SAMPLES'):
                batch_data = self._call_func('sample_from_dynamics_env')
                self._call_func('train_algo_from_synthesized_data', batch_data=batch_data)

            if self.time_step_func() - self.parameters(
                    'TEST_ALGO_EVERY_REAL_SAMPLE_COUNT') >= self._last_test_algo_point and self.time_step_func() > self.parameters(
                'START_TEST_ALGO_AFTER_SAMPLE_COUNT'):
                self._last_test_algo_point = self.time_step_func()
                self._call_func('test_algo')

            if self.time_step_func() - self.parameters(
                    'TEST_DYNAMICS_EVERY_REAL_SAMPLE_COUNT') >= self._last_test_dynamics_point and self.time_step_func() > self.parameters(
                'START_TEST_DYNAMICS_AFTER_SAMPLE_COUNT'):
                self._last_test_algo_point = self.time_step_func()
                self._call_func('test_dynamics')

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
