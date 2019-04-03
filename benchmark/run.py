from importlib import import_module

a = import_module('baconian.common.noise')
b = a.NormalActionNoise()
