class BaconianError(Exception):
    pass


class GlobalNameExistedError(BaconianError):
    pass


class StatusInfoNotRegisteredError(BaconianError):
    pass


class DynamicsNextStepOutputBoundError(BaconianError):
    pass


class MemoryBufferLessThanBatchSizeError(BaconianError):
    pass


class InappropriateParameterSetting(BaconianError):
    pass
