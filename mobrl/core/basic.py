import typeguard as tg
from mobrl.config.global_config import GlobalConfig
from mobrl.core.status import Status
from mobrl.core.util import register_name_globally


class Basic(object):
    """ Basic class within the whole framework"""
    STATUS_LIST = GlobalConfig.DEFAULT_BASIC_STATUS_LIST
    INIT_STATUS = GlobalConfig.DEFAULT_BASIC_INIT_STATUS
    required_key_list = ()

    def __init__(self, name: str, status=None, ):
        if not status:
            self._status = Status(self)
        else:
            self._status = status
        self._name = name
        register_name_globally(name=name, obj=self)

    def init(self):
        raise NotImplementedError

    def get_status(self) -> dict:
        return self._status.get_status()

    def set_status(self, val):
        self._status.set_status(val)

    @property
    def name(self):
        return self._name

    @property
    def status_list(self):
        return self.STATUS_LIST
