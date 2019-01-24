from src.core.pipeline import Pipeline
from src.core.global_config import GlobalConfig
from src.envs.env import Env
from src.core.config import Config
from src.rl.algo.algo import ModelBasedAlgo


class ModelBasedPipeline(Pipeline):
    STATE_LIST = ['state_not_inited', 'state_inited', 'state_algo_testing', 'state_algo_training', 'state_ended']
    INIT_STATE = 'state_not_inited'

    def __init__(self, config: Config, algo: ModelBasedAlgo, env: Env):
        self.algo = algo
        self.env = env

        super().__init__(config=config, init_state=self.INIT_STATE, states=self.STATE_LIST)

    def launch(self):
        assert self.is_state_not_inited()
        try:
            self.trigger('state_inited')
            while self.is_state_ended() is not True:
                self.trigger('state_training')
                self.trigger('state_testing')
                self.trigger('state_ended')
        except GlobalConfig.DEFAULT_CATCHED_EXCEPTION_OR_ERROR_LIST as e:
            self.trigger('state_corrupted')

    def on_enter_state_inited(self):
        pass

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

    def _is_flow_ended(self):
        # todo define different end condition here
        return False
