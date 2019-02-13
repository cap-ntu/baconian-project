from mbrl.common.sampler.sample_data import TransitionData
from mbrl.envs import make
from mbrl.envs.env_spec import EnvSpec
import numpy as np
from mbrl.test.tests.testSetup import BaseTestCase


class TestSampleData(BaseTestCase):
    def test_transition_data(self):
        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        a = TransitionData(env_spec)
        st = env.reset()
        for i in range(100):
            ac = env_spec.action_space.sample()
            st_new, re, done, _ = env.step(action=ac)
            a.append(state=st, new_state=st_new, action=ac, done=done, reward=re)
        self.assertEqual(a.reward_set.shape[0], 100)
        self.assertEqual(a.reward_set.shape[1], 1)
        self.assertEqual(a.done_set.shape[0], 100)
        self.assertEqual(a.done_set.shape[1], 1)

        self.assertEqual(a.action_set.shape[0], 100)
        self.assertEqual(a.state_set.shape[0], 100)
        self.assertEqual(a.new_state_set.shape[0], 100)

        iterator = a.return_generator()
        count = 0
        for st, new_st, ac, reward, terminal in iterator:
            count += 1
            self.assertTrue(env_spec.action_space.contains(ac))
            self.assertTrue(env_spec.obs_space.contains(st))
            self.assertTrue(env_spec.obs_space.contains(new_st))
            self.assertTrue(np.isscalar(reward))
            self.assertTrue(isinstance(terminal, bool))
        self.assertEqual(count, 100)

        a = TransitionData(obs_shape=list(np.array(env_spec.obs_space.sample()).shape),
                           action_shape=list(np.array(env_spec.action_space.sample()).shape))
        st = env.reset()
        for i in range(100):
            ac = env_spec.action_space.sample()
            st_new, re, done, _ = env.step(action=ac)
            a.append(state=st, new_state=st_new, action=ac, done=done, reward=re)
        self.assertEqual(a.reward_set.shape[0], 100)
        self.assertEqual(a.reward_set.shape[1], 1)
        self.assertEqual(a.done_set.shape[0], 100)
        self.assertEqual(a.done_set.shape[1], 1)

        self.assertEqual(a.action_set.shape[0], 100)
        self.assertEqual(a.state_set.shape[0], 100)
        self.assertEqual(a.new_state_set.shape[0], 100)

        iterator = a.return_generator()
        count = 0
        for st, new_st, ac, reward, terminal in iterator:
            count += 1
            self.assertTrue(env_spec.action_space.contains(ac))
            self.assertTrue(env_spec.obs_space.contains(st))
            self.assertTrue(env_spec.obs_space.contains(new_st))
            self.assertTrue(np.isscalar(reward))
            self.assertTrue(isinstance(terminal, bool))
        self.assertEqual(count, 100)

    def test_trajectory_data(self):
        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
