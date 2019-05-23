from baconian.algo.policy.policy import DeterministicPolicy
from baconian.core.core import EnvSpec
from baconian.core.parameters import Parameters
from baconian.config.global_config import GlobalConfig
from baconian.config.dict_config import DictConfig
from baconian.common.misc import *
from copy import deepcopy


class ConstantActionPolicy(DeterministicPolicy):
    required_key_dict = DictConfig.load_json(file_path=GlobalConfig().DEFAULT_CONSTANT_ACTION_POLICY_REQUIRED_KEY_LIST)

    def __init__(self, env_spec: EnvSpec, config_or_config_dict: (DictConfig, dict), name='policy'):
        config = construct_dict_config(config_or_config_dict, self)
        parameters = Parameters(parameters=dict(),
                                source_config=config)
        assert env_spec.action_space.contains(x=config('ACTION_VALUE'))
        super().__init__(env_spec, parameters, name)
        self.config = config

    def forward(self, *args, **kwargs):
        action = self.env_spec.action_space.unflatten(self.parameters('ACTION_VALUE'))
        assert self.env_spec.action_space.contains(x=action)
        return action

    def copy_from(self, obj) -> bool:
        super().copy_from(obj)
        self.parameters.copy_from(obj.parameters)
        return True

    def make_copy(self, *args, **kwargs):
        return ConstantActionPolicy(env_spec=self.env_spec,
                                    config_or_config_dict=deepcopy(self.config),
                                    *args, **kwargs)
