import unittest
import tensorflow as tf
from mbrl.tf.util import create_new_tf_session
from mbrl.common.util.recorder import global_recorder
from mbrl.common.util.logger import global_logger, global_console_logger
from mbrl.config.global_config import GlobalConfig
import shutil
import os


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestTensorflowSetup(BaseTestCase):
    default_id = 0

    def setUp(self):
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


class TestWithLogSet(BaseTestCase):
    def setUp(self):
        try:
            shutil.rmtree(GlobalConfig.DEFAULT_LOG_PATH)
        except FileNotFoundError:
            pass
        os.mkdir(GlobalConfig.DEFAULT_LOG_PATH)
        global_logger.init(config_or_config_dict=GlobalConfig.DEFAULT_LOG_CONFIG_DICT,
                           log_path=GlobalConfig.DEFAULT_LOG_PATH,
                           log_level=GlobalConfig.DEFAULT_LOG_LEVEL)
        global_console_logger.init(logger_name='console_logger',
                                   to_file_flag=True,
                                   level=GlobalConfig.DEFAULT_LOG_LEVEL,
                                   to_file_name=os.path.join(GlobalConfig.DEFAULT_LOG_PATH, 'console.log'))

    def tearDown(self):
        global_logger.reset()
        global_console_logger.reset()
