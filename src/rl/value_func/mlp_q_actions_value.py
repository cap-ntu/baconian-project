from src.core.parameters import Parameters
from src.envs.env_spec import EnvSpec
from src.rl.value_func.value_func import ValueFunction


class MLPQValueOnActions(ValueFunction):
    def __init__(self, env_spec: EnvSpec, parameters: Parameters):
        super().__init__(env_spec, parameters)

    def copy(self, obj) -> bool:
        return super().copy(obj)

    def forward(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass
