from overrides.overrides import overrides
from src.core.config import Config
from typeguard import typechecked
from transitions import Machine


class Pipeline(object):
    REQUIRED_KEY_LIST = []
    STATE_LIST = []
    INITE_STATE = None

    @typechecked
    def __init__(self, config: Config, init_state: str, states: list, transitions: (list, dict)):
        self.config = config
        self.finite_state_machine = Machine(self=self, transitions=transitions, states=states, initial=init_state)

    def launch(self, *args, **kwargs):
        raise NotImplementedError
