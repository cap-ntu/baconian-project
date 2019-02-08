from src.rl.algo.model_based.models.dynamics_model import DynamicsModel
from src.rl.algo.model_based.models.mlp_dynamics_model import ContinuousMLPDynamicsModel
import unittest
import unittest
from src.rl.algo.model_free import DQN
from src.envs.gym_env import make
from src.envs.env_spec import EnvSpec
from src.rl.value_func.mlp_q_value import MLPQValueFunction
import tensorflow as tf
from src.tf.util import create_new_tf_session
import numpy as np
from src.common.sampler.sample_data import TransitionData
from src.rl.policy.deterministic_mlp import DeterministicMLPPolicy


class TestDeterministicMLPPolicy(unittest.TestCase):
    def test_mlp_deterministic_policy(self):
        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
        tf.reset_default_graph()
        env = make('Swimmer-v1')
        env.reset()
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        sess = create_new_tf_session(cuda_device=0)

        policy = DeterministicMLPPolicy(env_spec=env_spec,
                                        name_scope='mlp_policy',
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
                                                "N_UNITS": env_spec.flat_action_dim,
                                                "TYPE": "DENSE",
                                                "W_NORMAL_STDDEV": 0.03
                                            }
                                        ],
                                        output_high=None,
                                        output_low=None,
                                        output_norm=None,
                                        input_norm=None,
                                        reuse=False)
        self.assertIsNotNone(tf.get_default_session())
        policy.init()
        for _ in range(10):
            ac = policy.forward(obs=env.observation_space.sample())
            self.assertTrue(env.action_space.contains(ac[0]))
        p2 = policy.make_copy(name_scope='test',
                              reuse=False)
        self.assertGreater(len(policy.parameters('tf_var_list')), 0)
        self.assertGreater(len(p2.parameters('tf_var_list')), 0)
        for var1, var2 in zip(policy.parameters('tf_var_list'), p2.parameters('tf_var_list')):
            self.assertEqual(var1.shape, var2.shape)
            self.assertNotEqual(id(var1), id(var2))

        p3 = policy.make_copy(name_scope='mlp_policy',
                              reuse=True)
        self.assertGreater(len(p3.parameters('tf_var_list')), 0)
        for var1, var2 in zip(policy.parameters('tf_var_list'), p3.parameters('tf_var_list')):
            self.assertEqual(var1.shape, var2.shape)
            self.assertEqual(id(var1), id(var2))
