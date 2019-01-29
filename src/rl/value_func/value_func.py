# Date: 11/16/18
# Author: Luke
# Project: ModelBasedRLFramework
from src.core.basic import Basic
import typeguard as tg
from src.core.parameters import Parameters
from src.envs.env_spec import EnvSpec


class ValueFunction(Basic):

    @tg.typechecked
    def __init__(self, env_spec: EnvSpec, parameters: Parameters):
        super().__init__()
        self.env_spec = env_spec
        self.parameters = parameters

    @property
    def obs_space(self):
        return self.env_spec.obs_space

    @property
    def action_space(self):
        return self.env_spec.action_space

    @tg.typechecked
    def copy(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        return True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def init(self, source_obj=None):
        raise NotImplementedError

    def make_copy(self, *args, **kwargs):
        raise NotImplementedError
