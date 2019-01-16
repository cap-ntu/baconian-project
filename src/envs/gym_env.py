from src.core.config import config
from src.envs.env import Env
from gym import make


class GymEnv(Env):
    def __init__(self, gym_env_id: str, config: config):
        super().__init__(config)
        self._gym_env = make(gym_env_id)

    def step(self, action):
        return self.unwrapped.step(action=action)

    def reset(self):
        return self.unwrapped.reset()

    def init(self):
        return self.reset()

    def seed(self, seed=None):
        return super().seed(seed)

    @property
    def unwrapped(self):
        return self._gym_env
