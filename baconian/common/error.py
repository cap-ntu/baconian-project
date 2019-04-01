class BaconianError(Exception):
    pass


class GlobalNameExistedError(BaconianError):
    pass


class StatusInfoNotRegisteredError(BaconianError):
    pass


class DynamicsNextStepOutputBoundError(BaconianError):
    pass
