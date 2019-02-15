from mbrl.core.basic import Basic
import typeguard as tg
from mbrl.core.parameters import Parameters
from mbrl.envs.env_spec import EnvSpec
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
    def copy_from(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        return True

    def make_copy(self, *args, **kwargs):
        raise NotImplementedError


# todo do we need this inheritance?
# class PlaceholderInputPolicy(Policy):
#     pass


class StochasticPolicy(Policy):
    @tg.typechecked
    def __init__(self, env_spec: EnvSpec, parameters: Parameters = None):
        super(StochasticPolicy, self).__init__(env_spec, parameters)
        self.state_input = None
        self.action_output = None

    def log_prob(self, *args, **kwargs):
        pass

    def prob(self, *args, **kwargs):
        pass

    def kl(self, other, *kwargs):
        pass

    def entropy(self, *args, **kwargs):
        pass

    def get_dist_info(self) -> tuple:
        pass


class MLPStochasticPolicy(Policy):

    def __init__(self, env_spec: EnvSpec, parameters: Parameters = None):
        super().__init__(env_spec, parameters)
        self.distribution_tensor_dict = None


class DeterministicPolicy(Policy):
    pass


if __name__ == '__main__':
    def test(*arg, **kwargs):
        print(arg, kwargs)


    test(1, 2, 3, s=1, t=2)
