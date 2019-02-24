import typeguard as tg
import abc
from typeguard import typechecked


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

    @typechecked
    def get_status(self) -> dict:
        return self()

    def append_new_info(self, info_key: str, init_value, under_status=None):
        if info_key == 'status':
            raise ValueError("can use key: status which is a system default key")
        if info_key in self._info_dict:
            return
        else:
            self._info_dict[info_key] = init_value

    def has_info(self, info_key):
        return info_key in self._info_dict

    def update_info(self, info_key, increment, under_status=None):
        if not self.has_info(info_key=info_key):
            self.append_new_info(info_key=info_key, init_value=0)
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

    def get_status(self) -> dict:
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
        if not self.has_info(info_key=info_key, under_status=under_status):
            self.append_new_info(info_key=info_key, init_value=0, under_status=under_status)
        self._info_dict_with_sub_info[under_status][info_key] += increment

    def reset(self):
        for key in self._status_list:
            self._info_dict_with_sub_info[key] = {}


def register_counter_info_to_status_decorator(increment, info_key, under_status=None):
    def wrap(fn):
        def wrap_with_self(self, *args, **kwargs):
            # todo call the fn first in order to get a correct status
            # todo a bug here, which is record() called in fn will lost the just appended info_key at the very first
            obj = self
            if not hasattr(obj, '_status') or not isinstance(getattr(obj, '_status'), StatusWithInfo):
                raise ValueError(
                    ' the object {} does not not have attribute StatusWithInfo instance or hold wrong type of Status'.format(
                        obj))

            assert isinstance(getattr(obj, '_status'), StatusWithInfo)
            obj_status = getattr(obj, '_status')
            # if isinstance(obj_status, StatusWithSubInfo):
            #     assert under_status
            if under_status:
                assert under_status in obj.STATUS_LIST
            obj_status.append_new_info(info_key=info_key, init_value=0, under_status=under_status)
            res = fn(self, *args, **kwargs)
            if under_status:
                assert under_status == obj.get_status()['status']
            obj_status.update_info(info_key=info_key, increment=increment, under_status=under_status)
            return res

        return wrap_with_self

    return wrap
