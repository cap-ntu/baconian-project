from mbrl.config.dict_config import DictConfig
from mbrl.core.pipelines.model_free_pipelines import ModelFreePipeline
from mbrl.envs.env import Env
from mbrl.agent.agent import Agent


class SampleCountBasedModelFreePipeline(ModelFreePipeline):
    def __init__(self, config: DictConfig, agent: Agent, env: Env):
        super().__init__(config, agent, env)
        self.total_train_sample_count = 0

    def launch(self):
        super().launch()

    def on_enter_state_inited(self):
        super().on_enter_state_inited()

    def on_exit_state_inited(self):
        super().on_exit_state_inited()

    def on_enter_state_testing(self):
        super().on_enter_state_testing()

    def on_exit_state_testing(self):
        super().on_exit_state_testing()

    def on_enter_state_training(self):
        super().on_enter_state_training()

    def on_exit_state_training(self):
        super().on_exit_state_training()

    def on_enter_state_ended(self):
        super().on_enter_state_ended()

    def on_exit_state_ended(self):
        super().on_exit_state_ended()

    def on_enter_state_corrupted(self):
        super().on_enter_state_corrupted()

    def on_exit_state_corrupted(self):
        super().on_exit_state_corrupted()

    def _is_flow_ended(self):
        pass
