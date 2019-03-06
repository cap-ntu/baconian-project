# import numpy as np
# import time
# import typeguard as tg
# import functools
#
#
# def random_snapshot(func):
#     @functools.wraps(func)
#     def wrapper(random_instance):
#         random_instance.state_snapshot = random_instance._np_random.get_state()
#         return func(random_instance)
#
#     return wrapper
#
#
# class Random(object):
#     """
#
#     A random utility that based on Numpy random module in order to better control the randomness of the system.
#     The python random is not recommended to use
#
#     """
#
#     @tg.typechecked
#     def __init__(self, seed: int = int(round(time.time() * 1000)) % (2 ** 32 - 1), global_state: bool = True):
#         self.seed = seed
#         if global_state:
#             self._np_random = np.random
#         else:
#             self._np_random = np.random.RandomState()
#         self._np_random.seed(seed)
#         self.state_snapshot = self._np_random.get_state()
#
#     @random_snapshot
#     def unwrapped(self):
#         return self._np_random
#
#     @tg.typechecked
#     def set_seed(self, seed: int):
#         self.seed = seed
#         self._np_random.seed(seed)
#
#     @tg.typechecked
#     def set_state(self, state: tuple):
#         self._np_random.set_state(state)
#
#     def reset_last_state(self):
#         self.set_state(self.state_snapshot)
#
#     def _register_all_np_method(self):
#         raise NotImplementedError
#
#
# if __name__ == '__main__':
#     r = Random()
#     print(r.unwrapped().rand(1, 1))
#     r.reset_last_state()
#     print(r.unwrapped().rand(1, 1))
