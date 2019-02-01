from src.config.dict_config import DictConfig
from typeguard import typechecked
from transitions import Machine


class Pipeline(object):
    REQUIRED_KEY_LIST = []
    STATE_LIST = []
    INIT_STATE = None

    @typechecked
    def __init__(self, config: DictConfig, init_state: str, states: list, transitions: (list, dict)):
        self.config = config
        self.finite_state_machine = Machine(model=self, transitions=transitions, states=states, initial=init_state)

    def launch(self, *args, **kwargs):
        raise NotImplementedError
