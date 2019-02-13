import unittest
from mbrl.envs import make
from mbrl.envs.env_spec import EnvSpec
from mbrl.rl.value_func.mlp_v_value import MLPVValueFunc
from mbrl.test.tests.test_setup import TestTensorflowSetup
import tensorflow as tf


class TestMLPVValueFunc(TestTensorflowSetup):
    def test_init(self):
        env = make('Swimmer-v1')

        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        mlp_v = MLPVValueFunc(env_spec=env_spec,
                              name_scope='mlp_q',
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
        # self.assertEqual(mlp_q.q_tensor.shape()[1], 1)
        mlp_v.init()

    def test_init_2(self):
        env = make('Swimmer-v1')

        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        state_input = tf.placeholder(shape=[None, env_spec.flat_obs_dim],
                                     dtype=tf.float32,
                                     name='state_ph')

        mlp_v = MLPVValueFunc(env_spec=env_spec,
                              name_scope='mlp_q',
                              state_input=state_input,
                              output_low=None,
                              output_high=None,
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
        mlp_v.init()
        mlp_v.forward(obs=env.observation_space.sample())


if __name__ == '__main__':
    unittest.main()
