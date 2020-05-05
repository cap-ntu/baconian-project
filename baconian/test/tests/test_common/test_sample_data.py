from baconian.common.sampler.sample_data import TransitionData, TrajectoryData
from baconian.envs.gym_env import make
from baconian.core.core import EnvSpec
import numpy as np
from baconian.test.tests.set_up.setup import BaseTestCase
from baconian.common.error import *


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
        self.assertEqual(a.done_set.shape[0], 100)
        self.assertEqual(a.action_set.shape[0], 100)
        self.assertEqual(a.state_set.shape[0], 100)
        self.assertEqual(a.new_state_set.shape[0], 100)

        self.assertEqual(a('reward_set').shape[0], 100)
        self.assertEqual(a('done_set').shape[0], 100)

        self.assertEqual(a('state_set').shape[0], 100)
        self.assertEqual(a('new_state_set').shape[0], 100)
        self.assertEqual(a('action_set').shape[0], 100)

        a = TransitionData(obs_shape=list(np.array(env_spec.obs_space.sample()).shape),
                           action_shape=list(np.array(env_spec.action_space.sample()).shape))
        st = env.reset()
        for i in range(100):
            ac = env_spec.action_space.sample()
            st_new, re, done, _ = env.step(action=ac)
            a.append(state=st, new_state=st_new, action=ac, done=done, reward=re)
        self.assertEqual(a.reward_set.shape[0], 100)
        self.assertEqual(a.done_set.shape[0], 100)

        self.assertEqual(a.action_set.shape[0], 100)
        self.assertEqual(a.state_set.shape[0], 100)
        self.assertEqual(a.new_state_set.shape[0], 100)

        self.assertEqual(a('reward_set').shape[0], 100)
        self.assertEqual(a('done_set').shape[0], 100)

        self.assertEqual(a('state_set').shape[0], 100)
        self.assertEqual(a('new_state_set').shape[0], 100)
        self.assertEqual(a('action_set').shape[0], 100)

        self.assertTrue(np.equal(a.get_mean_of('state_set'), a.apply_op('state_set', np.mean)).all())
        self.assertTrue(np.equal(a.get_sum_of('state_set'), a.apply_op('state_set', np.sum)).all())

        self.assertTrue(np.equal(a.get_sum_of('reward_set'), a.apply_op('reward_set', np.sum)).all())
        self.assertTrue(np.equal(a.get_sum_of('reward_set'), a.apply_op('reward_set', np.sum)).all())

        self.assertTrue(np.equal(a.get_sum_of('action_set'), a.apply_op('action_set', np.sum)).all())
        self.assertTrue(np.equal(a.get_sum_of('action_set'), a.apply_op('action_set', np.sum)).all())
        self.assertTrue(np.equal(a.apply_op('state_set', np.max, axis=-1), np.max(a('state_set'), axis=-1)).all())

        tmp_action = a('action_set').copy()
        a.apply_transformation(set_name='action_set', func=lambda x: x * 2, direct_apply=False)
        self.assertTrue(np.equal(tmp_action, a('action_set')).all())
        a.apply_transformation(set_name='action_set', func=lambda x: x * 2, direct_apply=True)
        self.assertTrue(np.equal(tmp_action * 2.0, a('action_set')).all())
        try:
            a.apply_transformation(set_name='action_set', func=lambda _: np.array([1, 2, 3]), direct_apply=True)
        except TransformationResultedToDifferentShapeError as e:
            pass
        else:
            raise TypeError

        a.apply_transformation(set_name='action_set', func=lambda x: x // 2, direct_apply=True)
        self.assertTrue(np.equal(tmp_action, a('action_set')).all())

        index = np.arange(len(a._internal_data_dict['state_set'][0])).tolist()
        b = a.get_copy()
        a.shuffle(index=list(index))
        for i in range(len(index)):
            for key in a._internal_data_dict.keys():
                self.assertTrue(np.equal(np.array(a._internal_data_dict[key][0][i]),
                                         np.array(b._internal_data_dict[key][0][i])).all())
        a.append_new_set(name='test', data_set=np.ones_like(a._internal_data_dict['state_set'][0]),
                         shape=a._internal_data_dict['state_set'][1])
        a.reset()
        self.assertEqual(a.reward_set.shape[0], 0)
        self.assertEqual(a.done_set.shape[0], 0)

        self.assertEqual(a.action_set.shape[0], 0)
        self.assertEqual(a.state_set.shape[0], 0)
        self.assertEqual(a.new_state_set.shape[0], 0)

        self.assertEqual(a('reward_set').shape[0], 0)
        self.assertEqual(a('done_set').shape[0], 0)

        self.assertEqual(a('state_set').shape[0], 0)
        self.assertEqual(a('new_state_set').shape[0], 0)
        self.assertEqual(a('action_set').shape[0], 0)

    def test_trajectory_data(self):
        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        a = TrajectoryData(env_spec)
        tmp_traj = TransitionData(env_spec)
        st = env.reset()
        re_list = []
        st_list = []
        for i in range(100):
            ac = env_spec.action_space.sample()
            st_new, re, done, _ = env.step(action=ac)
            st_list.append(st_new)
            re_list.append(re)
            if (i + 1) % 10 == 0:
                done = True
            else:
                done = False
            tmp_traj.append(state=st, new_state=st_new, action=ac, done=done, reward=re)
            if done:
                a.append(tmp_traj.get_copy())
                tmp_traj.reset()
        self.assertEqual(a.trajectories.__len__(), 10)
        for traj in a.trajectories:
            self.assertEqual(len(traj), 10)
