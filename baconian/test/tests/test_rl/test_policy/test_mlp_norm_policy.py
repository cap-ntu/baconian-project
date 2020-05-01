import unittest
from baconian.envs.gym_env import make
from baconian.core.core import EnvSpec
from baconian.algo.policy.normal_distribution_mlp import NormalDistributionMLPPolicy
from baconian.common.special import *
from baconian.test.tests.set_up.setup import TestTensorflowSetup


class TestNormalDistMLPPolicy(TestTensorflowSetup):
    def test_mlp_norm_dist_policy(self):
        env = make('Pendulum-v0')
        env.reset()
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        policy = NormalDistributionMLPPolicy(env_spec=env_spec,
                                             name='mlp_policy',
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
        dist_info = policy.get_dist_info()
        self.assertTrue(np.equal(dist_info[0]['shape'], policy.mean_output.shape.as_list()).all())
        self.assertTrue(np.equal(dist_info[1]['shape'], policy.logvar_output.shape.as_list()).all())
        for _ in range(10):
            ac = policy.forward(obs=env.observation_space.sample())
            self.assertTrue(env.action_space.contains(ac[0]))
        p2 = policy.make_copy(name='test',
                              name_scope='mlp_policy_2',
                              reuse=False)
        p2.init()
        self.assertGreater(len(policy.parameters('tf_var_list')), 0)
        self.assertGreater(len(p2.parameters('tf_var_list')), 0)
        for var1, var2 in zip(policy.parameters('tf_var_list'), p2.parameters('tf_var_list')):
            self.assertEqual(var1.shape, var2.shape)
            self.assertNotEqual(id(var1), id(var2))

        p3 = policy.make_copy(name='mlp_policy_ttt',
                              name_scope='mlp_policy',
                              reuse=True)
        p3.init()
        self.assertGreater(len(p3.parameters('tf_var_list')), 0)
        for var1, var2 in zip(policy.parameters('tf_var_list'), p3.parameters('tf_var_list')):
            self.assertEqual(var1.shape, var2.shape)
            self.assertEqual(id(var1), id(var2))

        # policy.copy_from(p2)]
        res_not_true = []
        for var1, var2, var3 in zip(policy.parameters('tf_var_list'), p2.parameters('tf_var_list'),
                                    p3.parameters('tf_var_list')):
            re1, re2, re3 = self.sess.run([var1, var2, var3])
            res_not_true.append(np.isclose(re1, re2).all())
            res_not_true.append(np.isclose(re3, re2).all())
            self.assertTrue(np.isclose(re1, re3).all())
        self.assertFalse(np.array(res_not_true).all())

        policy.copy_from(p2)

        for var1, var2, var3 in zip(policy.parameters('tf_var_list'), p2.parameters('tf_var_list'),
                                    p3.parameters('tf_var_list')):
            re1, re2, re3 = self.sess.run([var1, var2, var3])
            self.assertTrue(np.isclose(re1, re3).all())
            self.assertTrue(np.isclose(re2, re3).all())
            self.assertTrue(np.isclose(re1, re2).all())

    def test_func(self):
        env = make('Pendulum-v0')
        env.reset()
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        policy = NormalDistributionMLPPolicy(env_spec=env_spec,
                                             name='mlp_policy',
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
        print(
            policy.compute_dist_info(name='entropy',
                                     feed_dict={
                                         policy.state_input: make_batch(env_spec.obs_space.sample(),
                                                                        original_shape=env_spec.obs_shape)}))
        print(
            policy.compute_dist_info(name='prob',
                                     value=env_spec.action_space.sample(),
                                     feed_dict={
                                         policy.state_input: make_batch(env_spec.obs_space.sample(),
                                                                        original_shape=env_spec.obs_shape),
                                         policy.action_input: make_batch(env_spec.action_space.sample(),
                                                                         original_shape=env_spec.action_shape)}))
        new_policy = policy.make_copy(
            reuse=False,
            name='new_p',
            name_scope='mlp_policy_2',

        )
        new_policy.init()
        for var1, var2 in zip(policy.parameters('tf_var_list'), new_policy.parameters('tf_var_list')):
            print(var1.name)
            print(var2.name)
            self.assertNotEqual(var1.name, var2.name)
            self.assertNotEqual(id(var1), id(var2))
        obs1 = make_batch(env_spec.obs_space.sample(),
                          original_shape=env_spec.obs_shape,
                          )
        obs2 = make_batch(env_spec.obs_space.sample(),
                          original_shape=env_spec.obs_shape)
        kl1 = policy.compute_dist_info(name='kl', other=new_policy, feed_dict={
            policy.state_input: obs1,
            new_policy.state_input: obs2
        })
        kl2 = self.sess.run(policy.kl(other=new_policy), feed_dict={
            policy.state_input: obs1,
            new_policy.state_input: obs2
        })
        self.assertTrue(np.isclose(kl1, kl2).all())


if __name__ == '__main__':
    unittest.main()
