import unittest
from mbrl.test.tests.testSetup import BaseTestCase


class MyTestCase(BaseTestCase):
    def test_something(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
