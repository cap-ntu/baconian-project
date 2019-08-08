from baconian.core.flow.train_test_flow import Flow
from baconian.config.global_config import GlobalConfig
from baconian.common.logging import ConsoleLogger
from baconian.config.dict_config import DictConfig
from baconian.common.misc import *
from baconian.core.parameters import Parameters
from baconian.core.status import *


class MEPPO_Flow(Flow):
    """
    A typical flow for utilizing the model-based algo, it is not restricted to Dyna algorithms,
    but can be utilized by others.
    """

    required_func = ('train_algo', 'train_algo_from_synthesized_data', 'train_dynamics', 'test_algo', 'test_dynamics',
                     'sample_from_real_env', 'sample_from_dynamics_env', 'validate_policy_on_ensemble')

    required_key_dict = {
        "TEST_ALGO_EVERY_REAL_SAMPLE_COUNT": 1000,
        "TEST_DYNAMICS_EVERY_REAL_SAMPLE_COUNT": 1000,
        "TRAIN_ALGO_EVERY_REAL_SAMPLE_COUNT_FROM_REAL_ENV": 1000,
        "TRAIN_ALGO_EVERY_REAL_SAMPLE_COUNT_FROM_DYNAMICS_ENV": 1000,
        "TRAIN_DYNAMICS_EVERY_REAL_SAMPLE_COUNT": 1000,
        "START_TRAIN_ALGO_AFTER_SAMPLE_COUNT": 1,
        "START_TRAIN_DYNAMICS_AFTER_SAMPLE_COUNT": 1,
        "START_TEST_ALGO_AFTER_SAMPLE_COUNT": 1,
        "START_TEST_DYNAMICS_AFTER_SAMPLE_COUNT": 1,
        "WARM_UP_DYNAMICS_SAMPLES": 1000,
        "VALIDATION_THRESHOLD": 0.7,
        "VALIDATION_EVERY_FICTITIOUS_SET": 5,
        "SAMPLE_BEYOND_STOP_IMPROVEMENT": 10,
    }

    def __init__(self,
                 train_sample_count_func,
                 config_or_config_dict: (DictConfig, dict),
                 func_dict: dict, ):
        super().__init__(func_dict)
        super(MEPPO_Flow, self).__init__(func_dict=func_dict)
        config = construct_dict_config(config_or_config_dict, obj=self)
        self.parameters = Parameters(source_config=config, parameters=dict())
        self.time_step_func = train_sample_count_func
        self._last_train_algo_point = -1
        self._start_train_algo_point_from_dynamics = -1
        self._last_test_algo_point = -1
        self._start_train_dynamics_point = -1
        self._last_test_dynamics_point = -1
        self._last_performance = 0
        self._last_chance = 0
        self._fictitious_set_count = 0
        assert callable(train_sample_count_func)

    def _launch(self) -> bool:

        while True:

            if self._is_ended() is True:
                break

            real_batch_data = self._call_func('sample_from_real_env')

            if self.time_step_func() - self._start_train_dynamics_point <= self.parameters(
                        'TRAIN_DYNAMICS_EVERY_REAL_SAMPLE_COUNT') and \
                    self.time_step_func() > self.parameters('START_TRAIN_DYNAMICS_AFTER_SAMPLE_COUNT'):
                self._last_train_algo_point = self.time_step_func()
                self._call_func('train_dynamics', batch_data=real_batch_data)

            if self.time_step_func() <= self.parameters('WARM_UP_DYNAMICS_SAMPLES'):
                continue

            else:
                while True:
                    if self.time_step_func() - self._start_train_algo_point_from_dynamics <= self.parameters(
                                'TRAIN_ALGO_EVERY_REAL_SAMPLE_COUNT_FROM_DYNAMICS_ENV') and \
                            self.time_step_func() > self.parameters(
                            'START_TRAIN_ALGO_AFTER_SAMPLE_COUNT'):
                        batch_data = self._call_func('sample_from_dynamics_env')
                        self._call_func('train_algo_from_synthesized_data', batch_data=batch_data)

                    else:
                        self._fictitious_set_count += 1
                        self._last_chance += 1
                        if self._fictitious_set_count >= self.parameters('VALIDATION_EVERY_FICTITIOUS_SET'):
                            performance = self._call_func('validate_policy_on_ensemble')

                            if performance < self.parameters('VALIDATION_THRESHOLD'):
                                if self._last_chance > self.parameters('SAMPLE_BEYOND_IMPROVEMENT'):
                                    self._last_chance = 0
                                    self._start_train_dynamics_point = self.time_step_func()
                                    break
                        self._start_train_algo_point_from_dynamics = self.time_step_func()
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


def create_meppo_flow(train_algo_func, train_algo_from_synthesized_data_func,
                      train_dynamics_func, test_algo_func, test_dynamics_func, sample_from_real_env_func,
                      sample_from_dynamics_env_func,
                      validate_policy_on_ensemble_func,
                      test_algo_every_real_sample_count,
                      test_dynamics_every_real_sample_count,
                      train_algo_every_real_sample_count_by_data_from_real_env,
                      train_algo_every_real_sample_count_by_data_from_dynamics_env,
                      train_dynamics_every_real_sample_count,
                      start_train_algo_after_sample_count,
                      start_train_dynamics_after_sample_count,
                      start_test_dynamics_after_sample_count,
                      start_test_algo_after_sample_count,
                      warm_up_dynamics_samples,
                      validation_threshold,
                      validation_every_fictitious_set,
                      sample_beyond_stop_improvement,
                      train_samples_counter_func=None):
    config_dict = dict(
        TRAIN_ALGO_EVERY_REAL_SAMPLE_COUNT_FROM_REAL_ENV=train_algo_every_real_sample_count_by_data_from_real_env,
        TRAIN_ALGO_EVERY_REAL_SAMPLE_COUNT_FROM_DYNAMICS_ENV=train_algo_every_real_sample_count_by_data_from_dynamics_env,
        TEST_ALGO_EVERY_REAL_SAMPLE_COUNT=test_algo_every_real_sample_count,
        TEST_DYNAMICS_EVERY_REAL_SAMPLE_COUNT=test_dynamics_every_real_sample_count,
        TRAIN_DYNAMICS_EVERY_REAL_SAMPLE_COUNT=train_dynamics_every_real_sample_count,
        START_TRAIN_ALGO_AFTER_SAMPLE_COUNT=start_train_algo_after_sample_count,
        START_TRAIN_DYNAMICS_AFTER_SAMPLE_COUNT=start_train_dynamics_after_sample_count,
        START_TEST_ALGO_AFTER_SAMPLE_COUNT=start_test_algo_after_sample_count,
        START_TEST_DYNAMICS_AFTER_SAMPLE_COUNT=start_test_dynamics_after_sample_count,
        WARM_UP_DYNAMICS_SAMPLES=warm_up_dynamics_samples,
        VALIDATION_THRESHOLD=validation_threshold,
        VALIDATION_EVERY_FICTITIOUS_SET=validation_every_fictitious_set,
        SAMPLE_BEYOND_STOP_IMPROVEMENT=sample_beyond_stop_improvement,
    )

    def return_func_dict(s_dict):
        return dict(func=s_dict[0],
                    args=s_dict[1],
                    kwargs=s_dict[2])

    func_dict = dict(
        train_algo=return_func_dict(train_algo_func),
        train_algo_from_synthesized_data=return_func_dict(train_algo_from_synthesized_data_func),
        train_dynamics=return_func_dict(train_dynamics_func),
        test_algo=return_func_dict(test_algo_func),
        test_dynamics=return_func_dict(test_dynamics_func),
        sample_from_real_env=return_func_dict(sample_from_real_env_func),
        sample_from_dynamics_env=return_func_dict(sample_from_dynamics_env_func),
        validate_policy_on_ensemble=return_func_dict(validate_policy_on_ensemble_func)
    )
    if train_samples_counter_func is None:
        def default_train_samples_counter_func():
            return get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT')

        train_samples_counter_func = default_train_samples_counter_func

    return MEPPO_Flow(config_or_config_dict=config_dict,
                      train_sample_count_func=train_samples_counter_func,
                      func_dict=func_dict)
