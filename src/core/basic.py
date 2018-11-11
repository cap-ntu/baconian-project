import typeguard as tg


class Basic(object):
    """ Basic class within the whole framework"""

    def __init__(self):
        self.log_flag = False
        self.log_content = None
        pass


class Status(object):

    def __init__(self, cls: Basic):
        self.cls = cls
        if hasattr(cls, 'INIT_STATUS'):
            self.status_val = cls.INIT_STATUS
        else:
            self.status_val = None

    def __call__(self, *args, **kwargs):

        return self.cls
