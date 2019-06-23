from baconian.envs.gym_env import GymEnv
from baconian.test.tests.set_up.setup import TestWithLogSet
from gym import make
import os
import platform
from pathlib import Path

_PLATFORM = platform.system()
try:
    _PLATFORM_SUFFIX = {
        "Linux": "linux",
        "Darwin": "macos",
        "Windows": "win64"
    }[_PLATFORM]
except KeyError:
    raise OSError("Unsupported platform: {}".format(_PLATFORM))

have_mujoco_flag = True
try:
    import mujoco_py
except Exception:
    have_mujoco_flag = False

if have_mujoco_flag:
    os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') \
                                    + ':' + str(Path.home()) + '/.mujoco/mujoco200_{}/bin'.format(_PLATFORM_SUFFIX)

    os.environ['MUJOCO_PY_MUJOCO_PATH'] = os.environ.get('MUJOCO_PY_MUJOCO_PATH', '') \
                                          + str(Path.home()) + '/.mujoco/mujoco200_{}'.format(_PLATFORM_SUFFIX)


class TestEnv(TestWithLogSet):
    def test_gym_env(self):
        if have_mujoco_flag:
            a = GymEnv('FetchPickAndPlace-v1')
            a.set_status('TRAIN')
            self.assertEqual(a.total_step_count_fn(), 0)
            self.assertEqual(a._last_reset_point, 0)
            a.init()
            a.seed(10)
            a.reset()
            self.assertEqual(a.total_step_count_fn(), 0)
            self.assertEqual(a._last_reset_point, 0)
            for i in range(1000):
                new_st, re, done, _ = a.step(action=a.action_space.sample())
                self.assertEqual(a.total_step_count_fn(), i + 1)
                if done is True:
                    a.reset()
                    self.assertEqual(a._last_reset_point, a.total_step_count_fn())
                    self.assertEqual(a._last_reset_point, i + 1)
