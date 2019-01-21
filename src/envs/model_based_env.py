from src.core.config import Config
from src.envs.env import Env


class ModelBasedEnv(Env):
    def __init__(self, config: Config):
        super().__init__(config)

    def step(self, action):
        super().step(action)

    def reset(self):
        super().reset()

    def init(self):
        return super().init()
