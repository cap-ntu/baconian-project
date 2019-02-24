from mobrl.envs.gym_env import make
from mobrl.envs.env_spec import EnvSpec
from mobrl.algo.rl.policy .deterministic_mlp import DeterministicMLPPolicy
from mobrl.test.tests.test_setup import TestTensorflowSetup


class TestDeterministicMLPPolicy(TestTensorflowSetup):
    def test_mlp_deterministic_policy(self):
        env = make('Swimmer-v1')
        env.reset()
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

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
        policy.init()
        for _ in range(10):
            ac = policy.forward(obs=env.observation_space.sample())
            self.assertTrue(env.action_space.contains(ac[0]))
        p2 = policy.make_copy(name_scope='test',
                              reuse=False)
        p2.init()
        self.assertGreater(len(policy.parameters('tf_var_list')), 0)
        self.assertGreater(len(p2.parameters('tf_var_list')), 0)
        for var1, var2 in zip(policy.parameters('tf_var_list'), p2.parameters('tf_var_list')):
            self.assertEqual(var1.shape, var2.shape)
            self.assertNotEqual(id(var1), id(var2))

        p3 = policy.make_copy(name_scope='mlp_policy',
                              reuse=True)
        p3.init()
        self.assertGreater(len(p3.parameters('tf_var_list')), 0)
        for var1, var2 in zip(policy.parameters('tf_var_list'), p3.parameters('tf_var_list')):
            self.assertEqual(var1.shape, var2.shape)
            self.assertEqual(id(var1), id(var2))
