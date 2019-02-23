import typeguard as tg
from mbrl.config.global_config import GlobalConfig
from mbrl.core.status import Status


class Basic(object):
    """ Basic class within the whole framework"""
    STATUS_LIST = GlobalConfig.DEFAULT_BASIC_STATUS_LIST
    INIT_STATUS = GlobalConfig.DEFAULT_BASIC_INIT_STATUS

    def __init__(self, status=None):
        if not status:
            self._status = Status(self)
        else:
            self._status = status

    def init(self):
        raise NotImplementedError

    def get_status(self) -> dict:
        return self._status.get_status()

    @property
    def name(self):
        raise NotImplementedError

    @property
    def status_list(self):
        return self.STATUS_LIST
