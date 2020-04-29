import abc
from baconian.config.global_config import GlobalConfig
from baconian.common.logging import ConsoleLogger
from baconian.config.dict_config import DictConfig
from baconian.common.misc import *
from baconian.core.parameters import Parameters
from baconian.core.status import *
from baconian.common.error import *


class Flow(object):
    """
    Interface of experiment flow module, it defines the workflow of the reinforcement learning experiments.
    """
    required_func = ()
    required_key_dict = dict()

    def __init__(self, func_dict):
        """
        Constructor for Flow.

        :param func_dict: the function and its arguments that will be called in the Flow
        :type func_dict: dict
        """
        self.func_dict = func_dict
        for key in self.required_func:
            if key not in func_dict:
                raise MissedConfigError('miss key {}'.format(key))

    def launch(self) -> bool:
        """
        Launch the flow until it finished or catch a system-allowed errors (e.g., out of GPU memory, to ensure the log will be saved safely).

        :return: True if the flow correctly executed and finished
        :rtype: bool
        """
        try:
            return self._launch()
        except GlobalConfig().DEFAULT_ALLOWED_EXCEPTION_OR_ERROR_LIST as e:
            ConsoleLogger().print('error', 'error {} occurred'.format(e))
            return False

    def _launch(self) -> bool:
        """
        Abstract method to be implemented by subclass for a certain workflow.

        :return: True if the flow correctly executed and finished
        :rtype: bool
        """
        raise NotImplementedError

    def _call_func(self, key, **extra_kwargs):
        """
        Call a function that is pre-defined in self.func_dict

        :param key: name of the function, e.g., train, test, sample.
        :type key: str
        :param extra_kwargs: some extra kwargs you may want to be passed in the function calling
        :return: actual return value of the called function if self.func_dict has such function otherwise None.
        :rtype:
        """

        if self.func_dict[key]:
            return self.func_dict[key]['func'](*self.func_dict[key]['args'],
                                               **extra_kwargs,
                                               **self.func_dict[key]['kwargs'])
        else:
            return None


class TrainTestFlow(Flow):
    """
    A typical sampling-trainning and testing workflow, that used by most of model-free/model-based reinforcement
    learning method. Typically, it repeat the sampling(saving to memory if off policy)->training(from memory if
    off-policy, from samples if on-policy)->test
    """
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
        """
        Constructor of TrainTestFlow

        :param train_sample_count_func: a function indicates how much training samples the agent has collected currently.
        :type train_sample_count_func: method
        :param config_or_config_dict: a Config or a dict should have the keys: (TEST_EVERY_SAMPLE_COUNT, TRAIN_EVERY_SAMPLE_COUNT, START_TRAIN_AFTER_SAMPLE_COUNT, START_TEST_AFTER_SAMPLE_COUNT)
        :type config_or_config_dict: Config or dict
        :param func_dict: function dict, holds the keys: 'sample', 'train', 'test'. each item in the dict as also should be a dict, holds the keys 'func', 'args', 'kwargs'
        :type func_dict: dict
        """
        super(TrainTestFlow, self).__init__(func_dict=func_dict)
        config = construct_dict_config(config_or_config_dict, obj=self)
        self.parameters = Parameters(source_config=config, parameters=dict())
        self.time_step_func = train_sample_count_func
        self.last_train_point = -1
        self.last_test_point = -1
        assert callable(train_sample_count_func)

    def _launch(self) -> bool:
        """
        Launch the flow until it finished or catch a system-allowed errors
        (e.g., out of GPU memory, to ensure the log will be saved safely).

        :return: True if the flow correctly executed and finished
        :rtype: bool
        """
        while True:
            self._call_func('sample')
            if self.time_step_func() - self.parameters('TRAIN_EVERY_SAMPLE_COUNT') >= self.last_train_point and \
                    self.time_step_func() > self.parameters('START_TRAIN_AFTER_SAMPLE_COUNT'):
                self.last_train_point = self.time_step_func()
                self._call_func('train')
            if self.time_step_func() - self.parameters('TEST_EVERY_SAMPLE_COUNT') >= self.last_test_point and \
                    self.time_step_func() > self.parameters('START_TEST_AFTER_SAMPLE_COUNT'):
                self.last_test_point = self.time_step_func()
                self._call_func('test')

            if self._is_ended() is True:
                break
        return True

    def _is_ended(self):
        """

        :return: True if an experiment is ended
        :rtype: bool
        """
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


def create_train_test_flow(test_every_sample_count, train_every_sample_count, start_train_after_sample_count,
                           start_test_after_sample_count, train_func_and_args, test_func_and_args, sample_func_and_args,
                           train_samples_counter_func=None):
    config_dict = dict(
        TEST_EVERY_SAMPLE_COUNT=test_every_sample_count,
        TRAIN_EVERY_SAMPLE_COUNT=train_every_sample_count,
        START_TRAIN_AFTER_SAMPLE_COUNT=start_train_after_sample_count,
        START_TEST_AFTER_SAMPLE_COUNT=start_test_after_sample_count,
    )

    def return_func_dict(s_dict):
        return dict(func=s_dict[0],
                    args=s_dict[1],
                    kwargs=s_dict[2])

    func_dict = dict(
        train=return_func_dict(train_func_and_args),
        test=return_func_dict(test_func_and_args),
        sample=return_func_dict(sample_func_and_args),
    )
    if train_samples_counter_func is None:
        def default_train_samples_counter_func():
            return get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT')

        train_samples_counter_func = default_train_samples_counter_func

    return TrainTestFlow(config_or_config_dict=config_dict,
                         train_sample_count_func=train_samples_counter_func,
                         func_dict=func_dict)
