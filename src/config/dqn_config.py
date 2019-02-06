from src.config.dict_config import DictConfig
from src.config.global_config import GlobalConfig


class DQNDictConfig(DictConfig):
    def __init__(self, config_dict=None, cls_name="DQN"):
        super().__init__(required_key_dict=self.load_json(file_path=GlobalConfig.DEFAULT_DQN_REQUIRED_KEY_LIST),
                         config_dict=config_dict,
                         cls_name=cls_name)
