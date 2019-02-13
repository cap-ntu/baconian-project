from mbrl.rl.algo.algo import ModelBasedAlgo
from mbrl.rl.algo.model_based.models.dynamics_model import DynamicsModel


class IntelTrainer(ModelBasedAlgo):

    def __init__(self, env_spec, dynamics_model: DynamicsModel):
        super().__init__(env_spec=env_spec, dynamics_model=dynamics_model)

    def init(self):
        super().init()

    def train(self, *arg, **kwargs) -> dict:
        return super().train(*arg, **kwargs)

    def test(self, *arg, **kwargs):
        super().test(*arg, **kwargs)

    def predict(self, *arg, **kwargs):
        pass

    def append_to_memory(self, *args, **kwargs):
        pass
