from baconian.common.logging import Recorder


class Tuner(object):
    """
    Auto hyper parameter tuning module, tobe done
    """
    def __init__(self):
        self.recorder = Recorder(default_obj=self)
