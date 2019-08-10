from baconian.config.global_config import GlobalConfig
from baconian.test.tests.set_up.setup import BaseTestCase
from baconian.common.error import *
from baconian.algo.dynamics.reward_func.reward_func import RandomRewardFunc
from baconian.core.core import Basic

class TestCore(BaseTestCase):
    def test_global_config(self):
        GlobalConfig().set_new_config(config_dict=dict(DEFAULT_BASIC_INIT_STATUS='test'))
        assert GlobalConfig().DEFAULT_BASIC_INIT_STATUS == 'test'

        GlobalConfig().freeze_flag = True
        try:
            GlobalConfig().set_new_config(config_dict=dict(DEFAULT_BASIC_INIT_STATUS='test'))
        except AttemptToChangeFreezeGlobalConfigError:
            pass
        else:
            raise TypeError

        try:
            GlobalConfig().set('DEFAULT_LOG_PATH', 'tmp')
        except AttemptToChangeFreezeGlobalConfigError:
            pass
        else:
            raise TypeError

        try:
            GlobalConfig().DEFAULT_LOG_PATH = 'tmp'
        except AttemptToChangeFreezeGlobalConfigError:
            pass
        else:
            raise TypeError
        GlobalConfig().unfreeze()

    def test_config(self):
        config, _ = self.create_dict_config()

    def test_name_register(self):
        a = RandomRewardFunc()
        self.assertTrue(a.allow_duplicate_name)
        b = RandomRewardFunc()
        a = Basic(name='s')
        try:
            b = Basic(name='s')
        except GlobalNameExistedError as e:
            pass
        else:
            raise NotCatchCorrectExceptionError()
