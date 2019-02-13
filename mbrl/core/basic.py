import typeguard as tg
from mbrl.config.global_config import GlobalConfig


class Basic(object):
    """ Basic class within the whole framework"""
    STATUS_LIST = GlobalConfig.DEFAULT_BASIC_STATUS_LIST
    INIT_STATUS = GlobalConfig.DEFAULT_BASIC_INIT_STATUS

    def __init__(self):
        self.log_flag = False
        self.log_content = None
        self.status = Status(self)

    def init(self):
        raise NotImplementedError


class Status(object):

    def __init__(self, obj: Basic):
        self.obj = obj
        self._status_val = None
        if hasattr(obj, 'STATUS_LIST'):
            self.status_list = obj.STATUS_LIST
        else:
            self.status_list = None
        if hasattr(obj, 'INIT_STATUS') and obj.INIT_STATUS is not None:
            self.set_status(new_status=obj.INIT_STATUS)
        else:
            self._status_val = None

    def __call__(self, *args, **kwargs):
        return self._status_val

    @tg.typechecked
    def set_status(self, new_status: str):
        if self.status_list:
            try:
                assert new_status in self.status_list
            except AssertionError as e:
                print("{} New status :{} not in the status list: {} ".format(e, new_status, self.status_list))
            self._status_val = new_status
        else:
            self._status_val = new_status

    def get_status(self):
        return self()
