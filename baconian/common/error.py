class BaconianError(Exception):
    pass


class GlobalNameExistedError(BaconianError):
    pass


class StatusInfoNotRegisteredError(BaconianError):
    pass


class StateOrActionOutOfBoundError(BaconianError):
    pass


class MemoryBufferLessThanBatchSizeError(BaconianError):
    pass


class InappropriateParameterSetting(BaconianError):
    pass


class DuplicatedRegisteredError(BaconianError):
    pass


class LogPathOrFileNotExistedError(BaconianError):
    pass


class NotCatchCorrectExceptionError(BaconianError):
    pass


class AttemptToChangeFreezeGlobalConfigError(BaconianError):
    pass


class MissedConfigError(BaconianError):
    pass


class TransformationResultedToDifferentShapeError(BaconianError):
    pass


class WrongValueRangeError(BaconianError):
    pass


class ShapeNotCompatibleError(BaconianError):
    pass


class EnvNotExistedError(BaconianError):
    pass


class LogItemNotExisted(BaconianError):
    pass
