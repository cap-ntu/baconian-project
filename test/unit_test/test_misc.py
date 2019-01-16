import unittest
from src.envs.env_spec import EnvSpec
from gym.spaces import *


class TestMisc(unittest.TestCase):

    def create_env_spec(self):
        env = EnvSpec(obs_space=Box(low=0.0,
                                    high=1.0,
                                    shape=[2]),
                      action_space=Box(low=0.0,
                                       high=1.0,
                                       shape=[2]))
        return env

    def test_env_spec(self):
        env = self.create_env_spec()
        self.assertEqual(env.flat_obs_dim, 2)
        self.assertEqual(env.flat_action_dim, 2)
