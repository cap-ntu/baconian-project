"""
For experiments, its functionality should include:
1. experiment and config set up
2. logging control
3. hyper-param tuning etc.
4. visualization
5. any related experiment utility
...
"""
from baconian.core.core import Basic
from baconian.common.logging import ConsoleLogger
from baconian.config.global_config import GlobalConfig
from baconian.core.tuner import Tuner
from baconian.core.util import init_func_arg_record_decorator
import tensorflow as tf
from typeguard import typechecked
from baconian.tf.util import create_new_tf_session
from baconian.core.core import Env
from baconian.core.agent import Agent
from baconian.common.logging import Recorder
from baconian.core.status import *
from baconian.core.flow.train_test_flow import Flow
from baconian.core.global_var import reset_all as reset_global_var
from baconian.common.logging import reset_logging


class Experiment(Basic):
    STATUS_LIST = ('CREATED', 'INITED', 'RUNNING', 'FINISHED', 'CORRUPTED')
    INIT_STATUS = 'CREATED'
    # required_key_dict = DictConfig.load_json(file_path=GlobalConfig().DEFAULT_EXPERIMENT_REQUIRED_KEY_LIST)
    required_key_dict = dict()

    @init_func_arg_record_decorator()
    @typechecked
    def __init__(self,
                 name: str,
                 agent: Agent,
                 env: Env,
                 flow: Flow,
                 tuner: Tuner = None,
                 register_default_global_status=True
                 ):

        """

        :param name: name of experiment
        :type name: str
        :param agent: agent of experiment
        :type agent: Agent
        :param env: environment of experiment
        :type env: Env
        :param flow: control flow to experiment
        :type flow: Flow
        :param tuner: hyper-parameter tuning method, currently in development
        :type tuner: Tuner
        :param register_default_global_status: register info key and status into global status collection
        :type register_default_global_status: bool
        """
        super().__init__(status=StatusWithSingleInfo(obj=self), name=name)
        self.agent = agent
        self.env = env
        self.tuner = tuner
        self.recorder = Recorder(flush_by_split_status=False, default_obj=self)
        # self.status_collector = StatusCollector()
        self.flow = flow
        if register_default_global_status is True:
            get_global_status_collect().register_info_key_status(obj=agent,
                                                                 info_key='predict_counter',
                                                                 under_status='TRAIN',
                                                                 return_name='TOTAL_AGENT_TRAIN_SAMPLE_COUNT')
            get_global_status_collect().register_info_key_status(obj=agent,
                                                                 info_key='sample_counter',
                                                                 under_status='TRAIN',
                                                                 return_name='TOTAL_AGENT_TRAIN_SAMPLE_FUNC_COUNT')

            get_global_status_collect().register_info_key_status(obj=agent,
                                                                 info_key='predict_counter',
                                                                 under_status='TEST',
                                                                 return_name='TOTAL_AGENT_TEST_SAMPLE_COUNT')
            get_global_status_collect().register_info_key_status(obj=agent,
                                                                 info_key='update_counter',
                                                                 under_status='TRAIN',
                                                                 return_name='TOTAL_AGENT_UPDATE_COUNT')
            get_global_status_collect().register_info_key_status(obj=env,
                                                                 info_key='step',
                                                                 under_status='TEST',
                                                                 return_name='TOTAL_ENV_STEP_TEST_SAMPLE_COUNT')
            get_global_status_collect().register_info_key_status(obj=env,
                                                                 info_key='step',
                                                                 under_status='TRAIN',
                                                                 return_name='TOTAL_ENV_STEP_TRAIN_SAMPLE_COUNT')

    def init(self):
        """ Create a new TensorFlow session, and set status to 'INITED'."""
        create_new_tf_session()
        self.agent.init()
        self.env.init()
        self.set_status('INITED')

    def run(self):
        """ Run the experiment, and set status to 'RUNNING'."""
        GlobalConfig().freeze()
        self.init()
        self.set_status('RUNNING')
        res = self.flow.launch()
        if res is False:
            self.set_status('CORRUPTED')
        else:
            self.set_status('FINISHED')
        self._exit()

    def _exit(self):
        """ Exit the experiment, reset global configurations and logging module."""
        sess = tf.get_default_session()
        if sess:
            sess.__exit__(None, None, None)
        tf.reset_default_graph()
        reset_global_status_collect()
        reset_logging()
        reset_global_var()
        GlobalConfig().unfreeze()

    def TOTAL_AGENT_UPDATE_COUNT(self):
        return get_global_status_collect()('TOTAL_AGENT_UPDATE_COUNT')

    def TOTAL_AGENT_TRAIN_SAMPLE_COUNT(self):
        return get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT')

    def TOTAL_AGENT_TEST_SAMPLE_COUNT(self):
        return get_global_status_collect()('TOTAL_AGENT_TEST_SAMPLE_COUNT')

    def TOTAL_ENV_STEP_TRAIN_SAMPLE_COUNT(self):
        return get_global_status_collect()('TOTAL_ENV_STEP_TRAIN_SAMPLE_COUNT')

    def TOTAL_ENV_STEP_TEST_SAMPLE_COUNT(self):
        return get_global_status_collect()('TOTAL_ENV_STEP_TEST_SAMPLE_COUNT')
