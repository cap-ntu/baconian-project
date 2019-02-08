from src.core.basic import Basic
import typeguard as tg
from src.core.parameters import Parameters
from src.envs.env_spec import EnvSpec
import abc


class Policy(Basic):

    @tg.typechecked
    def __init__(self, env_spec: EnvSpec, parameters: Parameters = None):
        super().__init__()
        self.env_spec = env_spec
        self.parameters = parameters

    @property
    def obs_space(self):
        return self.env_spec.obs_space

    @property
    def action_space(self):
        return self.env_spec.action_space

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @tg.typechecked
    def copy(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        return True


class StochasticPolicy(Policy):
    pass


class DeterministicPolicy(Policy):
    pass


if __name__ == '__main__':
    def test(*arg, **kwargs):
        print(arg, kwargs)


    test(1, 2, 3, s=1, t=2)
