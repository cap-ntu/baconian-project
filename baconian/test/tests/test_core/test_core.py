from baconian.config.global_config import GlobalConfig
from baconian.test.tests.set_up.setup import BaseTestCase


class TestCore(BaseTestCase):
    def test_global_config(self):
        GlobalConfig.set_new_config(config_dict=dict(DEFAULT_BASIC_INIT_STATUS='test'))
        assert GlobalConfig.DEFAULT_BASIC_INIT_STATUS == 'test'

    def test_config(self):
        config, _ = self.create_dict_config()
