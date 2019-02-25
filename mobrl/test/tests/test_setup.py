import unittest
import tensorflow as tf
from mobrl.tf.util import create_new_tf_session
from mobrl.config.global_config import GlobalConfig
import shutil
import os
from mobrl.common.util.logger import Logger, ConsoleLogger
from mobrl.core.global_var import reset_all, get_all


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        reset_all()
        for key, val in get_all().items():
            self.assertTrue(len(val) == 0)

    def tearDown(self):
        reset_all()


class TestTensorflowSetup(BaseTestCase):
    default_id = 0

    def setUp(self):
        BaseTestCase.setUp(self)
        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
        tf.reset_default_graph()
        print('set tf device as {}'.format(self.default_id))
        self.sess = create_new_tf_session(cuda_device=self.default_id)

    def tearDown(self):
        if self.sess.run(tf.report_uninitialized_variables()).shape[0] != 0:
            print('some variables are not uninitialized:')
            print(self.sess.run(tf.report_uninitialized_variables()))
            print(self.sess.run(tf.report_uninitialized_variables()).shape)
            raise AssertionError('some variables are not uninitialized')

        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.close()
        BaseTestCase.tearDown(self)


class TestWithLogSet(BaseTestCase):
    def setUp(self):
        BaseTestCase.setUp(self)
        try:
            shutil.rmtree(GlobalConfig.DEFAULT_LOG_PATH)
        except FileNotFoundError:
            pass
        os.mkdir(GlobalConfig.DEFAULT_LOG_PATH)
        self.assertFalse(ConsoleLogger().inited_flag)
        self.assertFalse(Logger().inited_flag)

        Logger().init(config_or_config_dict=GlobalConfig.DEFAULT_LOG_CONFIG_DICT,
                      log_path=GlobalConfig.DEFAULT_LOG_PATH,
                      log_level=GlobalConfig.DEFAULT_LOG_LEVEL)
        ConsoleLogger().init(logger_name='console_logger',
                             to_file_flag=True,
                             level=GlobalConfig.DEFAULT_LOG_LEVEL,
                             to_file_name=os.path.join(GlobalConfig.DEFAULT_LOG_PATH, 'console.log'))

        self.assertTrue(ConsoleLogger().inited_flag)
        self.assertTrue(Logger().inited_flag)

    def tearDown(self):
        # todo a little bug here if to call super()
        BaseTestCase.tearDown(self)
        Logger().reset()
        ConsoleLogger().reset()
        self.assertFalse(ConsoleLogger().inited_flag)
        self.assertFalse(Logger().inited_flag)


class TestWithAll(TestTensorflowSetup, TestWithLogSet):
    def setUp(self):
        TestTensorflowSetup.setUp(self)
        TestWithLogSet.setUp(self)

    def tearDown(self):
        TestWithLogSet.tearDown(self)
        TestTensorflowSetup.tearDown(self)


class SimpleTestSetup(BaseTestCase):
    def setUp(self):
        BaseTestCase.setUp(self)
        try:
            shutil.rmtree(GlobalConfig.DEFAULT_LOG_PATH)
        except FileNotFoundError:
            pass
        os.mkdir(GlobalConfig.DEFAULT_LOG_PATH)
        self.assertFalse(ConsoleLogger().inited_flag)
        self.assertFalse(Logger().inited_flag)

    def tearDown(self):
        BaseTestCase.tearDown(self)
