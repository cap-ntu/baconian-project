"""
Terminal/done signal functions in most of some model-based methods are tricky problems,
since model itself require a reward function from outside.

Most of the codebase tune the definition of the terminal function differed from original one
without explicit notifying which bring the difficult for users to tune the algorithms without the effect
brought by reward functions.

In Baconian, we try to clarify this part, make sure the user is well aware of the effect of such implementations.

This is a work in progress.
"""
from baconian.algo.dynamics.terminal_func.terminal_func import TerminalFunc, FixedEpisodeLengthTerminalFunc, \
    RandomTerminalFunc
