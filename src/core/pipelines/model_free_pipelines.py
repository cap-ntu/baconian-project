from src.core.global_config import GlobalConfig
from src.core.config import Config
from src.core.pipeline import Pipeline
from src.rl import ModelFreeAlgo
from src.envs.env import Env


class ModelFreePipeline(Pipeline):
    STATE_LIST = ['state_not_inited', 'state_inited', 'state_testing', 'state_training', 'state_ended',
                  'state_corrupted']
    INITE_STATE = 'state_not_inited'

    def __init__(self, config: Config, algo: ModelFreeAlgo, env: Env):
        transitions = []
        self.algo = algo
        self.env = env
        super().__init__(config=config, init_state=self.INITE_STATE, states=self.STATE_LIST, transitions=transitions)
        self.finite_state_machine.add_transition('init', 'state_not_inited', 'state_inited')
        self.finite_state_machine.add_transition('train',
                                                 ['state_algo_testing', 'state_inited', 'state_algo_training'],
                                                 'state_algo_training')
        self.finite_state_machine.add_transition('test',
                                                 ['state_algo_training', 'state_inited', 'state_algo_testing'],
                                                 'state_algo_testing')
        self.finite_state_machine.add_transition('end',
                                                 ['state_algo_training', 'state_algo_testing', 'state_inited',
                                                  'state_not_inited'],
                                                 'state_ended',
                                                 conditions='_is_flow_ended')

    def launch(self):
        assert self.is_state_not_inited()
        try:
            self.trigger('init')
            while self.is_state_ended() is not True:
                self.trigger('train')
                self.trigger('test')
                self.trigger('end')
        except GlobalConfig.DEFAULT_CATCHED_EXCEPTION_OR_ERROR_LIST as e:
            self.to_state_corrupted()

    def on_enter_state_inited(self):
        self.algo.init()
        self.env.init()
        print('model-free pipeline finish inited')

    def on_exit_state_inited(self):
        pass

    def on_enter_state_testing(self):
        pass

    def on_exit_state_testing(self):
        pass

    def on_enter_state_training(self):
        pass

    def on_exit_state_training(self):
        pass

    def on_enter_state_ended(self):
        pass

    def on_exit_state_ended(self):
        pass

    def on_enter_state_corrupted(self):
        pass

    def on_exit_state_corrupted(self):
        pass
