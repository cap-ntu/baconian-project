from src.core.basic import Basic
from gym.core import Space
import src.util.util as util
import typeguard as tg


class Policy(Basic):

    @tg.typechecked
    def __init__(self, obs_space: Space, action_space: Space):
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space

    def sample(self):
        pass
