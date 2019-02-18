import unittest
import tensorflow as tf
from mbrl.tf.util import create_new_tf_session


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
