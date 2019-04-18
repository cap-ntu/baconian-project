import numpy as np
import tensorflow as tf
from baconian.tf.tf_parameters import ParametersWithTensorflowVariable
from baconian.config.dict_config import DictConfig
from baconian.core.core import Basic, EnvSpec
from baconian.config.global_config import GlobalConfig
from baconian.tf.util import create_new_tf_session
from baconian.algo.rl.model_free.dqn import DQN
from baconian.envs.gym_env import make
from baconian.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from baconian.test.tests.set_up.setup import TestTensorflowSetup, TestWithAll


class TestTensorflowParameters(TestWithAll):
    def test_tf_param(self):
        param, _ = self.create_tf_parameters()
        param.init()
        param.save_snapshot()
        param.load_snapshot()

    def test_save_load(self):

        param, _ = self.create_tf_parameters('param')

        param.init()
        var_val = [self.sess.run(var) for var in param('tf_var_list')]

        param_other, _ = self.create_tf_parameters(name='other_param')
        param_other.init()

        for i in range(10):
            param.save(sess=self.sess,
                       save_path=GlobalConfig().DEFAULT_LOG_PATH + '/model',
                       global_step=i)

        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
        tf.reset_default_graph()
        print('set tf device as {}'.format(self.default_id))
        self.sess = create_new_tf_session(cuda_device=self.default_id)

        param2, _ = self.create_tf_parameters('param')
        param2.init()
        param2.load(path_to_model=GlobalConfig().DEFAULT_LOG_PATH + '/model', global_step=9)
        for var1, var2 in zip(var_val, param2('tf_var_list')):
            self.assertTrue(np.equal(var1, self.sess.run(var2)).all())
