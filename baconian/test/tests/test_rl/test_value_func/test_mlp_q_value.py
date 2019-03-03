import unittest
from baconian.common.special import *
from baconian.test.tests.set_up.setup import TestTensorflowSetup


class TestMLPQValueFunction(TestTensorflowSetup):

    def test_init(self):
        mlp_q, local = self.create_mlp_q_func(name='mlp_q')
        env = local['env']
        env_spec = local['env_spec']
        mlp_q.init()

        action = env.action_space.sample()
        action = np.array([action])
        action = flatten_n(env_spec.action_space, action)
        mlp_q.forward(obs=env.observation_space.sample(), action=action)

        action = env.action_space.sample()
        action = np.array([action])
        action = flatten_n(space=env.action_space, obs=action)

        mlp_q.forward(obs=env.observation_space.sample(), action=action)

        action = env.action_space.sample()
        action = np.array([action])
        action = flatten_n(space=mlp_q.env_spec.action_space,
                           obs=make_batch(action, original_shape=mlp_q.env_spec.action_shape))

        mlp_q.forward(obs=env.observation_space.sample(), action=action)

    def test_copy(self):
        mlp_q, local = self.create_mlp_q_func(name='mlp_q')
        mlp_q.init()

        new_mlp = mlp_q.make_copy(name='new_mlp',
                                  name_scope='mlp_q',
                                  reuse=True)

        new_mlp.init()

        self.assertGreater(len(mlp_q.parameters('tf_var_list')), 0)
        self.assertGreater(len(new_mlp.parameters('tf_var_list')), 0)

        for var1, var2 in zip(mlp_q.parameters('tf_var_list'), new_mlp.parameters('tf_var_list')):
            self.assertEqual(var1.shape, var2.shape)
            self.assertEqual(id(var1), id(var2))

        not_reuse_mlp = mlp_q.make_copy(name='no-reuse-mlp',
                                        name_scope='mlp_no_reuse',
                                        reuse=False)
        not_reuse_mlp.init()
        self.assertGreater(len(not_reuse_mlp.parameters('tf_var_list')), 0)

        for var1, var2 in zip(mlp_q.parameters('tf_var_list'), not_reuse_mlp.parameters('tf_var_list')):
            self.assertEqual(var1.shape, var2.shape)
            self.assertNotEqual(id(var1), id(var2))


if __name__ == '__main__':
    unittest.main()
