from baconian.envs.dmcontrol_env import have_mujoco_flag
import unittest


class TestEnv(unittest.TestCase):
    def test_dmcontrol_env(self):
        if have_mujoco_flag:
            from baconian.envs.dmcontrol_env import DMControlEnv
            a = DMControlEnv('cartpole', 'swingup')
            a.set_status('TRAIN')
            self.assertEqual(a.total_step_count_fn(), 0)
            self.assertEqual(a._last_reset_point, 0)
            a.init()
            a.seed(10)
            a.reset()
            self.assertEqual(a.total_step_count_fn(), 0)
            self.assertEqual(a._last_reset_point, 0)
            for i in range(1000):
                new_st, re, done, _ = a.step(action=a.action_space.sample())
                self.assertEqual(a.total_step_count_fn(), i + 1)
                if done is True:
                    a.reset()
                    self.assertEqual(a._last_reset_point, a.total_step_count_fn())
                    self.assertEqual(a._last_reset_point, i + 1)
