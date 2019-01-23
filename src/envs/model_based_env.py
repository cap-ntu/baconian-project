import src.envs.env as env
import src.core.config as config


class ModelBasedEnv(env.Env):
    def __init__(self, config: config.Config):
        super().__init__(config)

    def step(self, action):
        super().step(action)

    def reset(self):
        super().reset()

    def init(self):
        return super().init()
