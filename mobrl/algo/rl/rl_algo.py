from mobrl.algo.algo import Algo
from mobrl.algo.rl.model_based.models.dynamics_model import DynamicsModel
from mobrl.core.core import EnvSpec


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
        self.dynamics_model = dynamics_model
