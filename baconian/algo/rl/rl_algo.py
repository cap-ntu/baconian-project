from baconian.algo.algo import Algo
from baconian.algo.dynamics.dynamics_model import DynamicsModel
from baconian.core.core import EnvSpec


class ModelFreeAlgo(Algo):
    def __init__(self, env_spec: EnvSpec, name: str = 'model_free_algo'):
        super(ModelFreeAlgo, self).__init__(env_spec, name)


class OnPolicyAlgo(Algo):
    pass


class OffPolicyAlgo(Algo):
    pass


class ValueBasedAlgo(Algo):
    pass


class PolicyBasedAlgo(Algo):
    pass


class ModelBasedAlgo(Algo):
    def __init__(self, env_spec, dynamics_model: DynamicsModel, name: str = 'model_based_algo'):
        super(ModelBasedAlgo, self).__init__(env_spec, name)
        self._dynamics_model = dynamics_model

    def train_dynamics(self, *args, **kwargs):
        pass

    def test_dynamics(self, *args, **kwargs):
        pass
