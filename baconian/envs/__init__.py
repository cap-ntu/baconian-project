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

# TODO potential bug here if mujoco py change default path in the future

os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') \
                                + ':' + str(Path.home()) + '/.mujoco/mujoco200/bin'

if os.environ.get('MUJOCO_PY_MUJOCO_PATH', '') == ' ':
    os.environ['MUJOCO_PY_MUJOCO_PATH'] = str(Path.home()) + '/.mujoco/mujoco200'

if os.environ.get('MUJOCO_PY_MJKEY_PATH', '') == ' ':
    os.environ['MUJOCO_PY_MJKEY_PATH'] = str(Path.home()) + '/.mujoco/mujoco200'

os.environ['MJKEY_PATH'] = os.environ.get('MUJOCO_PY_MJKEY_PATH', '')
os.environ['MJLIB_PATH'] = os.environ.get('MUJOCO_PY_MUJOCO_PATH', '')

# TODO disable the rendering temporarily
os.environ['DISABLE_MUJOCO_RENDERING'] = '1'

have_mujoco_flag = True
try:
    from dm_control import mujoco
    from gym.envs.mujoco import mujoco_env
    from dm_control import suite

    from dm_control.rl.specs import ArraySpec
    from dm_control.rl.specs import BoundedArraySpec

    from collections import OrderedDict
except Exception:
    have_mujoco_flag = False
