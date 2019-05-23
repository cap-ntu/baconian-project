import unittest
from baconian.envs.gym_env import make
from baconian.core.core import EnvSpec
from baconian.algo.value_func import MLPVValueFunc
from baconian.test.tests.set_up.setup import TestTensorflowSetup
import tensorflow as tf


class TestMLPVValueFunc(TestTensorflowSetup):
    def test_init(self):
        env = make('Pendulum-v0')

        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        state_input = tf.placeholder(shape=[None, env_spec.flat_obs_dim],
                                     dtype=tf.float32,
                                     name='state_ph')

        mlp_v = MLPVValueFunc(env_spec=env_spec,
                              name_scope='mlp_q',
                              name='mlp_q',
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

    def test_copy(self):
        env = make('Pendulum-v0')

        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        state_input = tf.placeholder(shape=[None, env_spec.flat_obs_dim],
                                     dtype=tf.float32,
                                     name='state_ph')

        mlp_v = MLPVValueFunc(env_spec=env_spec,
                              name_scope='mlp_v',
                              name='mlp_v',
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

        new_mlp = mlp_v.make_copy(name='new_mlp',
                                  name_scope='mlp_v',
                                  reuse=True)

        new_mlp.init()

        self.assertGreater(len(mlp_v.parameters('tf_var_list')), 0)
        self.assertGreater(len(new_mlp.parameters('tf_var_list')), 0)

        for var1, var2 in zip(mlp_v.parameters('tf_var_list'), new_mlp.parameters('tf_var_list')):
            self.assertEqual(var1.shape, var2.shape)
            self.assertEqual(id(var1), id(var2))

        not_reuse_mlp = mlp_v.make_copy(name='no-reuse-mlp',
                                        name_scope='mlp_no_reuse',
                                        reuse=False)
        not_reuse_mlp.init()
        self.assertGreater(len(not_reuse_mlp.parameters('tf_var_list')), 0)

        for var1, var2 in zip(mlp_v.parameters('tf_var_list'), not_reuse_mlp.parameters('tf_var_list')):
            self.assertEqual(var1.shape, var2.shape)
            self.assertNotEqual(id(var1), id(var2))


if __name__ == '__main__':
    unittest.main()
