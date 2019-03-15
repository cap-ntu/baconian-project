from baconian.core.core import EnvSpec
from baconian.test.tests.set_up.setup import TestTensorflowSetup
from baconian.envs.gym_env import make


class TestDeterministicMLPPolicy(TestTensorflowSetup):
    def test_mlp_deterministic_policy(self):
        env = make('Pendulum-v0')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        policy, locals = self.create_mlp_deterministic_policy(name='mlp_policy', env_spec=env_spec)
        policy.init()
        for _ in range(10):
            ac = policy.forward(obs=env.observation_space.sample())
            self.assertTrue(env.action_space.contains(ac[0]))
        p2 = policy.make_copy(name='test',
                              name_scope='test',
                              reuse=False)
        p2.init()
        self.assertGreater(len(policy.parameters('tf_var_list')), 0)
        self.assertGreater(len(p2.parameters('tf_var_list')), 0)
        for var1, var2 in zip(policy.parameters('tf_var_list'), p2.parameters('tf_var_list')):
            self.assertEqual(var1.shape, var2.shape)
            self.assertNotEqual(id(var1), id(var2))

        p3 = policy.make_copy(name='mlp_policy_2',
                              name_scope='mlp_policy',
                              reuse=True)
        p3.init()
        self.assertGreater(len(p3.parameters('tf_var_list')), 0)
        for var1, var2 in zip(policy.parameters('tf_var_list'), p3.parameters('tf_var_list')):
            self.assertEqual(var1.shape, var2.shape)
            self.assertEqual(id(var1), id(var2))