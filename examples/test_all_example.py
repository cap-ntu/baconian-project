import glob
import os
import importlib.util

ABS_PATH = os.path.dirname(os.path.realpath(__file__))


def test_all():
    file_list = glob.glob(os.path.join(ABS_PATH, '*.py'))
    file_list.remove(os.path.realpath(__file__))
    for f in file_list:
        spec = importlib.util.spec_from_file_location('', f)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


if __name__ == '__main__':
    test_all()
