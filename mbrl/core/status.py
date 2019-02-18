import typeguard as tg
import abc


class Status(object):

    def __init__(self, obj):
        self.obj = obj
        self._status_val = None
        if hasattr(obj, 'STATUS_LIST'):
            self._status_list = obj.STATUS_LIST
        else:
            self._status_list = None
        if hasattr(obj, 'INIT_STATUS') and obj.INIT_STATUS is not None:
            self.set_status(new_status=obj.INIT_STATUS)
        else:
            self._status_val = None

    def __call__(self, *args, **kwargs):
        return dict(status=self._status_val)

    @tg.typechecked
    def set_status(self, new_status: str):
        if self._status_list:
            try:
                assert new_status in self._status_list
            except AssertionError as e:
                print("{} New status :{} not in the status list: {} ".format(e, new_status, self._status_list))
            self._status_val = new_status
        else:
            self._status_val = new_status

    def get_status(self) -> dict:
        return self()


class StatusWithInfo(Status):
    @abc.abstractmethod
    def append_new_info(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def has_info(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def update_info(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError


class StatusWithSingleInfo(StatusWithInfo):
    # todo StatusWithInfo
    def __init__(self, obj):
        super().__init__(obj)

        self._info_dict = {}

    def __call__(self, *args, **kwargs):
        res = super().__call__(*args, **kwargs)
        return {**res, **self._info_dict}

    def set_status(self, new_status: str):
        return super().set_status(new_status)

    def get_status(self):
        return self()

    def append_new_info(self, info_key: str, init_value):
        if info_key == 'status':
            raise ValueError("can use key: status which is a system default key")
        if info_key in self._info_dict:
            return
        else:
            self._info_dict[info_key] = init_value

    def has_info(self, info_key):
        return info_key in self._info_dict

    def update_info(self, info_key, increment):
        assert self.has_info(info_key=info_key)
        self._info_dict[info_key] += increment

    def reset(self):
        self._info_dict = {}


class StatusWithSubInfo(StatusWithInfo):

    def __init__(self, obj):
        super().__init__(obj)
        if not hasattr(obj, 'STATUS_LIST') or not hasattr(obj, 'INIT_STATUS'):
            raise ValueError(
                "StatusWithSubInfo require the source object to have class attr: STATUS_LIST and INIT_STATUS")

        self._info_dict_with_sub_info = {}
        for key in self._status_list:
            self._info_dict_with_sub_info[key] = {}

    def __call__(self, *args, **kwargs):
        res = super().__call__(*args, **kwargs)
        return {**res, **self._info_dict_with_sub_info[self._status_val]}

    def set_status(self, new_status: str):
        return super().set_status(new_status)

    def get_status(self):
        return self()

    def append_new_info(self, info_key: str, init_value, under_status=None):
        if not under_status:
            under_status = self._status_val
        if info_key == 'status':
            raise ValueError("can use key: status which is a system default key")
        if info_key in self._info_dict_with_sub_info[under_status]:
            return
        else:
            self._info_dict_with_sub_info[under_status][info_key] = init_value

    def has_info(self, info_key, under_status=None):
        if not under_status:
            under_status = self._status_val
        return info_key in self._info_dict_with_sub_info[under_status]

    def update_info(self, info_key, increment, under_status=None):
        if not under_status:
            under_status = self._status_val
        assert self.has_info(info_key=info_key, under_status=under_status)
        self._info_dict_with_sub_info[under_status][info_key] += increment

    def reset(self):
        for key in self._status_list:
            self._info_dict_with_sub_info[key] = {}


def register_counter_status_decorator(increment, key):
    def wrap(fn):
        def wrap_with_self(self, *args, **kwargs):
            # todo call the fn first in order to get a correct status
            res = fn(self, *args, **kwargs)
            obj = self
            if not hasattr(obj, 'status') or not isinstance(getattr(obj, 'status'), StatusWithInfo):
                raise ValueError(
                    'in order to count the calling time, the object {} did not have attribute StatusWithInfo instance or with wrong type of Status'.format(
                        obj))
            assert isinstance(getattr(obj, 'status'), StatusWithInfo)
            obj_status = getattr(obj, 'status')
            if not obj_status.has_info(info_key=key):
                obj_status.append_new_info(info_key=key, init_value=0)
            obj_status.update_info(info_key=key, increment=increment)

            return res

        return wrap_with_self

    return wrap
