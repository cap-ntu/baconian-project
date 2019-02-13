import unittest
from mbrl.envs import make
from mbrl.envs.env_spec import EnvSpec
from mbrl.rl.value_func.mlp_v_value import MLPVValueFunc
import tensorflow as tf
from mbrl.tf.util import create_new_tf_session


class TestMLPVValueFunc(BaseTestCase):
    def test_init(self):

        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
            # sess.close()
        tf.reset_default_graph()

        env = make('Swimmer-v1')
        sess = create_new_tf_session(cuda_device=0)

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
        if tf.get_default_session():
            tf.get_default_session().close()

    def test_init_2(self):

        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
            # sess.close()
        tf.reset_default_graph()

        env = make('Swimmer-v1')
        sess = create_new_tf_session(cuda_device=0)

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
