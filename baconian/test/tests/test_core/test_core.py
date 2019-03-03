from baconian.config.global_config import GlobalConfig
from baconian.test.tests.set_up.setup import BaseTestCase


class TestCore(BaseTestCase):
    def test_global_config(self):
        GlobalConfig.set_new_config(config_dict=dict(DEFAULT_BASIC_INIT_STATUS='test'))
        assert GlobalConfig.DEFAULT_BASIC_INIT_STATUS == 'test'

        GlobalConfig.set(key='DEFAULT_LOG_PATH', val='/home/dls/CAP/mobrl-internal/mobrl/test/tests/tmp_path2')
        self.assertEqual(GlobalConfig.DEFAULT_LOG_PATH, '/home/dls/CAP/mobrl-internal/mobrl/test/tests/tmp_path2')

    def test_config(self):
        config, _ = self.create_dict_config()
