from baconian.envs.gym_env import make
from baconian.core.core import EnvSpec
from baconian.test.tests.set_up.setup import BaseTestCase
from baconian.algo.misc.replay_buffer import UniformRandomReplayBuffer


class TestReplaybuffer(BaseTestCase):
    def test_transition_data(self):
        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        a = UniformRandomReplayBuffer(limit=10000, action_shape=env_spec.action_shape,
                                      observation_shape=env_spec.obs_shape)
        st = env.reset()
        for i in range(100):
            ac = env_spec.action_space.sample()
            st_new, re, done, _ = env.step(action=ac)
            a.append(obs0=st, obs1=st_new, action=ac, reward=re, terminal1=done)
            st = st_new
        batch = a.sample(batch_size=10)
        self.assertTrue(batch.state_set.shape[0] == 10)
        self.assertTrue(batch.action_set.shape[0] == 10)
        self.assertTrue(batch.reward_set.shape[0] == 10)
        self.assertTrue(batch.done_set.shape[0] == 10)
        self.assertTrue(batch.new_state_set.shape[0] == 10)
