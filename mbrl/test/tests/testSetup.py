import unittest
import tensorflow as tf
import sys
import os

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
print('join {} into environ path'.format(path))
src_dir = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
sys.path.append(src_dir)
print('join {} into environ path'.format(src_dir))

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
            # sess.exit(None, None, None)
            sess.close()
        tf.reset_default_graph()
        print('set tf device as {}'.format(self.default_id))
        self.sess = create_new_tf_session(cuda_device=self.default_id)

    def tearDown(self):
        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.close()
