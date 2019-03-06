from baconian.core.core import Basic, Env, EnvSpec
from baconian.common.sampler.sampler import Sampler
from baconian.config.global_config import GlobalConfig
from baconian.algo.algo import Algo
from typeguard import typechecked
from baconian.algo.rl.misc.epsilon_greedy import ExplorationStrategy
from baconian.common.sampler.sample_data import SampleData
from baconian.common.logging import Recorder, record_return_decorator
from baconian.core.status import StatusWithSubInfo
from baconian.core.status import register_counter_info_to_status_decorator
from baconian.core.util import init_func_arg_record_decorator
from baconian.config.dict_config import DictConfig
from baconian.common.misc import *
from baconian.common.logging import ConsoleLogger
from baconian.common.sampler.sample_data import TransitionData, TrajectoryData
from baconian.common.schedules import EventSchedule


class Agent(Basic):
    STATUS_LIST = ['NOT_INIT', 'JUST_INITED', 'TRAIN', 'TEST']
    INIT_STATUS = 'NOT_INIT'
    required_key_dict = DictConfig.load_json(GlobalConfig.DEFAULT_AGENT_REQUIRED_KEY_LIST)

    @init_func_arg_record_decorator()
    @typechecked
    def __init__(self, name,
                 config_or_config_dict: (DictConfig, dict),
                 env: Env,
                 algo: Algo, env_spec: EnvSpec,
                 sampler: Sampler = None,
                 exploration_strategy: ExplorationStrategy = None,
                 algo_saving_scheduler: EventSchedule = None):
        super(Agent, self).__init__(name=name, status=StatusWithSubInfo(self))
        self.config = construct_dict_config(config_or_config_dict, obj=self)
        self.total_test_samples = 0
        self.total_train_samples = 0
        self.env = env
        self.algo = algo
        self._env_step_count = 0
        self.sampler = sampler
        self.recorder = Recorder()
        if exploration_strategy:
            assert isinstance(exploration_strategy, ExplorationStrategy)
            self.explorations_strategy = exploration_strategy
        self.sampler = sampler if sampler else Sampler(env_spec=env_spec, name='{}_sampler'.format(name))

        self.algo_saving_scheduler = algo_saving_scheduler

    @record_return_decorator(which_recorder='self')
    @register_counter_info_to_status_decorator(increment=1, info_key='update_counter', under_status='TRAIN')
    def train(self):
        self.set_status('TRAIN')
        ConsoleLogger().print('info', 'agent enter training')

        res = self.sample(env=self.env,
                          sample_count=self.config('TRAIN_SAMPLES_COUNT'),
                          store_flag=True,
                          in_test_flag=False)
        self.total_train_samples += len(res)
        self.algo.train()
        ConsoleLogger().print('info', "Mean reward_func is {}".format(res.get_mean_of(set_name='reward_set')))
        if self.algo_saving_scheduler and self.algo_saving_scheduler.value() is True:
            self.algo.save(global_step=self._status.get_specific_info_key_status(info_key='update_counter',
                                                                                 under_status='TRAIN'))
        return dict(average_test_reward=res.get_mean_of(set_name='reward_set'))

    @record_return_decorator(which_recorder='self')
    def test(self):
        self.set_status('TEST')
        ConsoleLogger().print('info', 'agent enter testing')

        res = self.sample(env=self.env,
                          sample_count=self.config('TEST_SAMPLES_COUNT'),
                          store_flag=False,
                          in_test_flag=True)
        self.total_test_samples += len(res)
        ConsoleLogger().print('info', "Mean reward_func is {}".format(res.get_mean_of(set_name='reward_set')))
        return dict(average_test_reward=res.get_mean_of(set_name='reward_set'))

    @register_counter_info_to_status_decorator(increment=1, info_key='predict_counter', under_status=('TRAIN', 'TEST'),
                                               ignore_wrong_status=True)
    def predict(self, in_test_flag, **kwargs):
        if in_test_flag:
            self.set_status('TEST')
        else:
            self.set_status('TRAIN')
        if self.explorations_strategy and not in_test_flag:
            return self.explorations_strategy.predict(**kwargs, algo=self.algo)
        else:
            return self.algo.predict(**kwargs)

    @register_counter_info_to_status_decorator(increment=1, info_key='sample_counter', under_status=('TRAIN', 'TEST'),
                                               ignore_wrong_status=True)
    def sample(self, env, sample_count: int, in_test_flag: bool, store_flag=False) -> (TransitionData, TrajectoryData):
        if in_test_flag:
            self.set_status('TEST')
            env.set_status('TEST')
        else:
            self.set_status('TRAIN')
            env.set_status('TRAIN')
        batch_data = self.sampler.sample(agent=self,
                                         env=env,
                                         in_test_flag=in_test_flag,
                                         sample_count=sample_count)
        if store_flag is True:
            self.store_samples(samples=batch_data)
        return batch_data

    def init(self):
        self.algo.init()
        self.set_status('JUST_INITED')

    @typechecked
    def store_samples(self, samples: SampleData):
        self.algo.append_to_memory(samples=samples)
