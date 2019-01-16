import typeguard as tg


class Basic(object):
    """ Basic class within the whole framework"""
    STATUS_LIST = None
    INIT_STATUS = None

    def __init__(self):
        self.log_flag = False
        self.log_content = None
        self.status = Status(self)

    def init(self):
        raise NotImplementedError


class Status(object):

    def __init__(self, obj: Basic):
        self.obj = obj
        if hasattr(obj, 'INIT_STATUS'):
            self.status_val = obj.INIT_STATUS
        else:
            self.status_val = None
        if hasattr(obj, 'STATUS_LIST'):
            self.status_dict = obj.STATUS_LIST
        else:
            self.status_dict = None

    def __call__(self, *args, **kwargs):
        return self.status_val

    @tg.typechecked
    def set_status(self, new_status: str):
        if self.status_dict:
            assert new_status in self.status_dict
            self.status_val = new_status
        else:
            self.status_val = new_status

    def get_status(self):
        return self()
