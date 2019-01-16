import unittest
import sys
import os

print(sys.path)
from gym import make
from src.envs.env_spec import EnvSpec
from src.rl.value_func.mlp_q_value import MLPQValueFunction


class TestMLPQValueFunction(unittest.TestCase):
    def test_init(self):
        env = make('Swimmer-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        mlp_q = MLPQValueFunction(env_spec=env_spec,
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
        self.assertEqual(mlp_q.q_tensor.shape()[1], 1)
        mlp_q.init()

    def test_run_time(self):
        pass
