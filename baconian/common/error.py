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


class TransformationResultedToDifferentShape(BaconianError):
    pass
