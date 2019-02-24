from mobrl.config.global_config import GlobalConfig
from mobrl.test.tests.test_setup import BaseTestCase


class TestCore(BaseTestCase):
    def test_global_config(self):
        GlobalConfig.set_new_config(config_dict=dict(DEFAULT_BASIC_INIT_STATUS='test'))
        assert GlobalConfig.DEFAULT_BASIC_INIT_STATUS == 'test'
