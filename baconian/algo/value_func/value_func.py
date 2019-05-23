from baconian.core.core import Basic, EnvSpec
import typeguard as tg
from baconian.core.parameters import Parameters
import abc


class ValueFunction(Basic):

    @tg.typechecked
    def __init__(self, env_spec: EnvSpec, parameters: Parameters = None, name='value_func'):
        super().__init__(name)
        self.env_spec = env_spec
        self.parameters = parameters

    @property
    def obs_space(self):
        return self.env_spec.obs_space

    @property
    def action_space(self):
        return self.env_spec.action_space

    @tg.typechecked
    def copy_from(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        return True

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @tg.typechecked
    def update(self, param_update_args: dict, *args, **kwargs):
        self.parameters.update(**param_update_args)

    @abc.abstractmethod
    def init(self, source_obj=None):
        raise NotImplementedError

    @abc.abstractmethod
    def make_copy(self, *args, **kwargs):
        raise NotImplementedError


class VValueFunction(ValueFunction):
    def __init__(self, state_input, env_spec: EnvSpec, parameters: Parameters = None, name='value_func'):
        self.state_input = state_input
        super().__init__(env_spec, parameters, name)


class QValueFunction(ValueFunction):
    def __init__(self, state_input, action_input, env_spec: EnvSpec, parameters: Parameters = None, name='value_func'):
        self.state_input = state_input
        self.action_input = action_input
        super().__init__(env_spec, parameters, name)
