from src.core.pipeline import Pipeline
from src.core.global_config import GlobalConfig
from src.envs.env import Env
from src.config.dict_config import DictConfig
from src.rl.algo.algo import ModelBasedAlgo


class ModelBasedPipeline(Pipeline):
    """
    This class implement a very naive model based rl pipeline which is:
    train dynamics model -> train agent -> test agent -> train dynamics model ... -> end
    """
    STATE_LIST = ['state_not_inited', 'state_inited', 'state_agent_testing', 'state_agent_training', 'state_ended',
                  'state_dynamics_testing', 'state_dynamics_training']
    INIT_STATE = 'state_not_inited'

    def __init__(self, config: DictConfig, algo: ModelBasedAlgo, env: Env):
        self.algo = algo
        self.env = env

        super().__init__(config=config, init_state=self.INIT_STATE, states=self.STATE_LIST)
        # todo move the hard code here
        self.finite_state_machine.add_transition('init', 'state_not_inited', 'state_inited')
        self.finite_state_machine.add_transition('train agent',
                                                 ['state_agent_testing', 'state_inited', 'state_agent_training',
                                                  'state_dynamics_testing', 'state_dynamics_training'],
                                                 'state_agent_training')
        self.finite_state_machine.add_transition('test agent',
                                                 ['state_agent_training', 'state_inited', 'state_agent_testing',
                                                  'state_dynamics_testing', 'state_dynamics_training'],
                                                 'state_agent_testing')

        self.finite_state_machine.add_transition('train dyanmics',
                                                 ['state_agent_testing', 'state_inited', 'state_agent_training',
                                                  'state_dynamics_testing', 'state_dynamics_training'],
                                                 'state_dynamics_training')
        self.finite_state_machine.add_transition('test dyanmics',
                                                 ['state_agent_training', 'state_inited', 'state_agent_testing',
                                                  'state_dynamics_testing', 'state_dynamics_training'],
                                                 'state_dynamics_testing')

        self.finite_state_machine.add_transition('end',
                                                 ['state_agent_training', 'state_agent_testing', 'state_inited',
                                                  'state_not_inited'],
                                                 'state_ended',
                                                 conditions='_is_flow_ended')

    def launch(self):
        assert self.is_state_not_inited()
        try:
            self.trigger('init')
            while self.is_state_ended() is False:
                self.trigger('train dynamics')
                self.trigger('test dynamics')
                self.trigger('train agent')
                self.trigger('test agent')
                self.trigger('end')
        except GlobalConfig.DEFAULT_CATCHED_EXCEPTION_OR_ERROR_LIST as e:
            self.to_state_corrupted()

    def on_enter_state_inited(self):
        pass

    def on_exit_state_inited(self):
        pass

    def on_enter_state_agent_testing(self):
        pass

    def on_exit_state_agent_testing(self):
        pass

    def on_enter_state_agent_training(self):
        pass

    def on_exit_state_agent_training(self):
        pass

    def on_enter_state_dynamics_testing(self):
        pass

    def on_exit_state_dynamics_testing(self):
        pass

    def on_enter_state_dynamics_training(self):
        pass

    def on_exit_state_dynamics_training(self):
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
