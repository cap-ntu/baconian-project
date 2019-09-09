import os
import shutil

import numpy as np
import tensorflow as tf

from baconian.common.logging import Logger, ConsoleLogger
from baconian.config.global_config import GlobalConfig
from baconian.core.global_var import reset_all, get_all
from baconian.tf.util import create_new_tf_session
from baconian.test.tests.set_up.class_creator import ClassCreatorSetup
from baconian.core.status import reset_global_status_collect


class BaseTestCase(ClassCreatorSetup):
    def setUp(self):
        reset_all()
        reset_global_status_collect()
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
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.default_id)
        self.sess = create_new_tf_session()

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

    def assert_var_list_equal(self, var_list1, var_list2):
        for var1, var2 in zip(var_list1, var_list2):
            res1, res2 = self.sess.run([var1, var2])
            self.assertTrue(np.equal(res1, res2).all())

    def assert_var_list_at_least_not_equal(self, var_list1, var_list2):
        res = []
        for var1, var2 in zip(var_list1, var_list2):
            res1, res2 = self.sess.run([var1, var2])
            res.append(np.equal(res1, res2).all())
        self.assertFalse(np.array(res).all())

    def assert_var_list_id_equal(self, var_list1, var_list2):
        for var1, var2 in zip(var_list1, var_list2):
            self.assertTrue(id(var1) == id(var2))

    def assert_var_list_id_no_equal(self, var_list1, var_list2):
        for var1, var2 in zip(var_list1, var_list2):
            self.assertTrue(id(var1) != id(var2))


class TestWithLogSet(BaseTestCase):
    def setUp(self):
        BaseTestCase.setUp(self)
        try:
            shutil.rmtree(GlobalConfig().DEFAULT_LOG_PATH)
        except FileNotFoundError:
            pass
        # os.makedirs(GlobalConfig().DEFAULT_LOG_PATH)
        # self.assertFalse(ConsoleLogger().inited_flag)
        # self.assertFalse(Logger().inited_flag)

        Logger().init(config_or_config_dict=GlobalConfig().DEFAULT_LOG_CONFIG_DICT,
                      log_path=GlobalConfig().DEFAULT_LOG_PATH,
                      log_level=GlobalConfig().DEFAULT_LOG_LEVEL)
        ConsoleLogger().init(logger_name='console_logger',
                             to_file_flag=True,
                             level=GlobalConfig().DEFAULT_LOG_LEVEL,
                             to_file_name=os.path.join(Logger().log_dir, 'console.log'))

        self.assertTrue(ConsoleLogger().inited_flag)
        self.assertTrue(Logger().inited_flag)

    def tearDown(self):
        Logger().reset()
        ConsoleLogger().reset()
        BaseTestCase.tearDown(self)
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
            shutil.rmtree(GlobalConfig().DEFAULT_LOG_PATH)
        except FileNotFoundError:
            pass
        os.makedirs(GlobalConfig().DEFAULT_LOG_PATH)
        self.assertFalse(ConsoleLogger().inited_flag)
        self.assertFalse(Logger().inited_flag)

    def tearDown(self):
        BaseTestCase.tearDown(self)
