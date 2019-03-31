from unittest import TestLoader, TextTestRunner, TestSuite
import sys
import os

path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(path)
# print('join {} into environ path'.format(path))
src_dir = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
sys.path.append(src_dir)
print('join {} into environ path'.format(src_dir))


def test_all(dir=''):
    loader = TestLoader()
    suite = TestSuite()
    for all_test_suite in loader.discover(start_dir=os.path.join(path, 'tests', dir), pattern='test*.py'):
        for test_case in all_test_suite:
            suite.addTest(test_case)
    TextTestRunner().run(test=suite)


if __name__ == '__main__':
    test_all()
