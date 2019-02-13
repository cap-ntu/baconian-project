from mbrl.config.global_config import GlobalConfig
from mbrl.test.tests.testSetup import BaseTestCase


class TestCore(BaseTestCase):
    def test_global_config(self):
        GlobalConfig.set_new_config(config_dict=dict(DEFAULT_BASIC_INIT_STATUS='test'))
        assert GlobalConfig.DEFAULT_BASIC_INIT_STATUS == 'test'
