from mobrl.envs.gym_env import make
from mobrl.envs.env_spec import EnvSpec
from mobrl.common.sampler.sample_data import TransitionData, TrajectoryData
from mobrl.test.tests.set_up.setup import TestTensorflowSetup
import tensorflow as tf


class TestPPO(TestTensorflowSetup):
    def test_init(self):
        ppo, locals = self.create_ppo()
        env = locals['env']
        env_spec = locals['env_spec']
        ppo.init()
        print(tf.report_uninitialized_variables())
        data = TransitionData(env_spec)
        st = env.reset()
        for i in range(100):
            ac = ppo.predict(st)
            assert ac.shape[0] == 1
            self.assertTrue(env_spec.action_space.contains(ac[0]))
            new_st, re, done, _ = env.step(ac)
            if i == 99:
                done = True
            data.append(state=st, new_state=new_st, action=ac, reward=re, done=done)
        ppo.append_to_memory(data)
        print(ppo.train())

        traj_data = TrajectoryData(env_spec=env_spec)
        traj_data.append(data)
        print(
            ppo.train(trajectory_data=traj_data,
                      train_iter=10,
                      sess=self.sess))
        # ppo.append_to_memory(data)
        # for i in range(1000):
        #     print(ppo.train())


if __name__ == '__main__':
    TestPPO()
