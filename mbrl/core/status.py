import typeguard as tg


class Status(object):

    def __init__(self, obj):
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
        return dict(status=self._status_val)

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


class StatusWithInfo(Status):
    # todo StatusWithInfo
    def __init__(self, obj):
        super().__init__(obj)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def set_status(self, new_status: str):
        return super().set_status(new_status)

    def get_status(self):
        return super().get_status()
