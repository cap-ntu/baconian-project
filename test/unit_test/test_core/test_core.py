import unittest
from src.core.global_config import GlobalConfig


class TestCore(unittest.TestCase):
    def test_global_config(self):
        GlobalConfig.set_new_config(config_dict=dict(DEFAULT_BASIC_INIT_STATUS='test'))
        assert GlobalConfig.DEFAULT_BASIC_INIT_STATUS == 'test'
