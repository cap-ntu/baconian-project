import numpy as np
import tensorflow as tf
from mobrl.tf.tf_parameters import TensorflowParameters
from mobrl.config.dict_config import DictConfig
from mobrl.core.core import Basic, EnvSpec
from mobrl.config.global_config import GlobalConfig
from mobrl.tf.util import create_new_tf_session
from mobrl.algo.rl.model_free.dqn import DQN
from mobrl.envs.gym_env import make
from mobrl.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from mobrl.test.tests.set_up.setup import TestTensorflowSetup, TestWithAll


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
                       save_path=GlobalConfig.DEFAULT_LOG_PATH + '/model',
                       global_step=i)

        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
        tf.reset_default_graph()
        print('set tf device as {}'.format(self.default_id))
        self.sess = create_new_tf_session(cuda_device=self.default_id)

        param2, _ = self.create_tf_parameters('param')
        param2.init()
        param2.load(path_to_model=GlobalConfig.DEFAULT_LOG_PATH + '/model', global_step=9)
        for var1, var2 in zip(var_val, param2('tf_var_list')):
            self.assertTrue(np.equal(var1, self.sess.run(var2)).all())

    def test_save_load_with_dqn(self):
        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        mlp_q = MLPQValueFunction(env_spec=env_spec,
                                  name_scope='mlp_q',
                                  name='mlp_q',
                                  mlp_config=[
                                      {
                                          "ACT": "RELU",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "1",
                                          "N_UNITS": 16,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      },
                                      {
                                          "ACT": "LINEAR",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "OUPTUT",
                                          "N_UNITS": 1,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      }
                                  ])
        dqn = DQN(env_spec=env_spec,
                  adaptive_learning_rate=True,
                  config_or_config_dict=dict(REPLAY_BUFFER_SIZE=1000,
                                             GAMMA=0.99,
                                             BATCH_SIZE=10,
                                             Q_NET_L1_NORM_SCALE=0.001,
                                             Q_NET_L2_NORM_SCALE=0.001,
                                             LEARNING_RATE=0.001,
                                             TRAIN_ITERATION=1,
                                             DECAY=0.5),
                  value_func=mlp_q)
        dqn.init()
