from baconian.core.core import Basic, Env, EnvSpec
from baconian.common.sampler.sampler import Sampler
from baconian.common.error import *
from baconian.algo.algo import Algo
from typeguard import typechecked
from baconian.algo.misc import ExplorationStrategy
from baconian.common.sampler.sample_data import SampleData
from baconian.common.logging import Recorder, record_return_decorator
from baconian.core.status import StatusWithSubInfo
from baconian.core.status import register_counter_info_to_status_decorator
from baconian.core.util import init_func_arg_record_decorator
from baconian.common.logging import ConsoleLogger
from baconian.common.sampler.sample_data import TransitionData, TrajectoryData
from baconian.common.schedules import EventScheduler
from baconian.common.noise import AgentActionNoiseWrapper
from baconian.core.parameters import Parameters


class Agent(Basic):
    STATUS_LIST = ('NOT_INIT', 'JUST_INITED', 'TRAIN', 'TEST')
    INIT_STATUS = 'NOT_INIT'
    required_key_dict = {}

    @init_func_arg_record_decorator()
    @typechecked
    def __init__(self, name,
                 # config_or_config_dict: (DictConfig, dict),
                 env: Env,
                 algo: Algo,
                 env_spec: EnvSpec,
                 sampler: Sampler = None,
                 noise_adder: AgentActionNoiseWrapper = None,
                 exploration_strategy: ExplorationStrategy = None,
                 algo_saving_scheduler: EventScheduler = None):
        """

        :param name: the name of the agent instance
        :type name: str
        :param env: environment that interacts with agent
        :type env: Env
        :param algo: algorithm of the agent
        :type algo: Algo
        :param env_spec: environment specifications: action apace and environment space
        :type env_spec: EnvSpec
        :param sampler: sampler
        :type sampler: Sampler
        :param noise_adder: add action noise for exploration in action space
        :type noise_adder: AgentActionNoiseWrapper
        :param exploration_strategy: exploration strategy in action space
        :type exploration_strategy: ExplorationStrategy
        :param algo_saving_scheduler: control the schedule the varying parameters in training process
        :type algo_saving_scheduler: EventSchedule
        """
        super(Agent, self).__init__(name=name, status=StatusWithSubInfo(self))
        self.parameters = Parameters(parameters=dict())
        self.total_test_samples = 0
        self.total_train_samples = 0
        self.env = env
        self.algo = algo
        self._env_step_count = 0
        self.sampler = sampler
        self.recorder = Recorder(default_obj=self)
        self.env_spec = env_spec
        if exploration_strategy:
            assert isinstance(exploration_strategy, ExplorationStrategy)
            self.explorations_strategy = exploration_strategy
        else:
            self.explorations_strategy = None
        self.sampler = sampler if sampler else Sampler(env_spec=env_spec, name='{}_sampler'.format(name))
        self.noise_adder = noise_adder
        self.algo_saving_scheduler = algo_saving_scheduler

    # @record_return_decorator(which_recorder='self')
    @register_counter_info_to_status_decorator(increment=1, info_key='update_counter', under_status='TRAIN')
    def train(self, *args, **kwargs):
        """
        train the agent

        :return: Only if memory buffer did not have enough data to train, return False
        :rtype: bool
        """
        self.set_status('TRAIN')
        self.algo.set_status('TRAIN')
        ConsoleLogger().print('info', 'train agent:')
        try:
            res = self.algo.train(*args, **kwargs)
        except MemoryBufferLessThanBatchSizeError as e:
            ConsoleLogger().print('warning', 'memory buffer did not have enough data to train, skip training')
            return False

        ConsoleLogger().print('info', res)

        if self.algo_saving_scheduler and self.algo_saving_scheduler.value() is True:
            self.algo.save(global_step=self._status.get_specific_info_key_status(info_key='update_counter',
                                                                                 under_status='TRAIN'))

    # @record_return_decorator(which_recorder='self')
    def test(self, sample_count, sample_trajectory_flag: bool = False):
        """
        test the agent

        :param sample_count: how many transitions, or, transitions in trajectories used to evaluate the agent's performance
        :type sample_count: int
        :param sample_trajectory_flag: True for sampling trajectory instead of transitions
        :type sample_count: bool
        """
        self.set_status('TEST')
        self.algo.set_status('TEST')
        ConsoleLogger().print('info', 'test: agent with {},sample_trajectory_flag {}'.format(sample_count,
                                                                                             sample_trajectory_flag))
        if sample_trajectory_flag is True:
            left_sample_count = sample_count
            while left_sample_count > 0:
                res = self.sample(env=self.env,
                                  sample_count=1,
                                  sample_type='trajectory',
                                  store_flag=False,
                                  in_which_status='TEST')
                self.total_test_samples += len(res)
                left_sample_count -= len(res)

        else:
            res = self.sample(env=self.env,
                              sample_count=sample_count,
                              sample_type='transition',
                              store_flag=False,
                              in_which_status='TEST')
            self.total_test_samples += len(res)

    @register_counter_info_to_status_decorator(increment=1, info_key='predict_counter', under_status=('TRAIN', 'TEST'),
                                               ignore_wrong_status=True)
    def predict(self, **kwargs):
        """
        predict the action given the state

        :param kwargs: rest parameters, include key: obs
        :return: predicted action
        :rtype: numpy ndarray
        """
        res = None
        if self.explorations_strategy and not self.is_testing:
            res = self.explorations_strategy.predict(**kwargs, algo=self.algo)
        else:
            if self.noise_adder and not self.is_testing:
                res = self.env_spec.action_space.clip(self.noise_adder(self.algo.predict(**kwargs)))
            else:
                res = self.algo.predict(**kwargs)
        self.recorder.append_to_obj_log(obj=self, attr_name='action', status_info=self.get_status(), log_val=res)
        return res

    @register_counter_info_to_status_decorator(increment=1, info_key='sample_counter', under_status=('TRAIN', 'TEST'),
                                               ignore_wrong_status=True)
    def sample(self, env, sample_count: int, in_which_status: str, store_flag=False,
               sample_type: str = 'transition') -> (
            TransitionData, TrajectoryData):
        """
        sample a certain number of data from environment

        :param env: environment to sample
        :param sample_count: int, sample count
        :param in_which_status: string, environment status
        :param store_flag: to store environment samples or not, default False
        :param sample_type: the type of sample, 'transition' by default
        :return: sample data from environment
        :rtype: some subclass of SampleData: TrajectoryData or TransitionData
        """
        self.set_status(in_which_status)
        env.set_status(in_which_status)
        self.algo.set_status(in_which_status)
        ConsoleLogger().print('info',
                              "agent sampled {} {} under status {}".format(sample_count, sample_type,
                                                                           self.get_status()))
        batch_data = self.sampler.sample(agent=self,
                                         env=env,
                                         sample_type=sample_type,
                                         in_which_status=in_which_status,
                                         sample_count=sample_count)
        if store_flag is True:
            self.store_samples(samples=batch_data)
        # todo when we have transition/ trajectory data here, the mean or sum results are still valid?
        ConsoleLogger().print('info',
                              "sample: mean reward {}, sum reward {}".format(
                                  batch_data.get_mean_of(set_name='reward_set'),
                                  batch_data.get_sum_of(set_name='reward_set')))
        self.recorder.append_to_obj_log(obj=self, attr_name='average_reward',
                                        status_info=self.get_status(),
                                        log_val=batch_data.get_mean_of('reward_set'))
        self.recorder.append_to_obj_log(obj=self, attr_name='sum_reward',
                                        status_info=self.get_status(),
                                        log_val=batch_data.get_sum_of('reward_set'))
        return batch_data

    def init(self):
        """
        Initialize the algorithm, and set status to 'JUST_INITED'.
        """
        self.algo.init()
        self.set_status('JUST_INITED')

    @typechecked
    def store_samples(self, samples: SampleData):
        """
        store the samples into memory/replay buffer if the algorithm that agent hold need to do so, like DQN, DDPG

        :param samples: sample data of the experiment
        :type samples: SampleData
        """
        self.algo.append_to_memory(samples=samples)

    @property
    def is_training(self):
        """
        Check whether the agent is training. Return a boolean value.

        :return: true if the agent is training
        :rtype: bool
        """
        return self.get_status()['status'] == 'TRAIN'

    @property
    def is_testing(self):
        """
        Check whether the agent is testing. Return a boolean value.

        :return: true if the agent is testing
        :rtype: bool
        """
        return self.get_status()['status'] == 'TEST'
