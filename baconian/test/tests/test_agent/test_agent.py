from baconian.test.tests.set_up.setup import TestWithAll
from baconian.common.sampler.sample_data import SampleData


class TestAgent(TestWithAll):
    def test_agent(self):
        algo, local = self.create_dqn()
        env = local['env']
        env_spec = local['env_spec']
        agent, _ = self.create_agent(algo=algo, env=env,
                                     env_spec=env_spec,
                                     eps=self.create_eps(env_spec=env_spec)[0])
        self.register_global_status_when_test(agent, env)
        agent.init()
        env.reset()
        data = agent.sample(env=env, sample_count=10, store_flag=False, in_which_status='TEST')
        self.assertTrue(isinstance(data, SampleData))
        self.assertEqual(agent.algo.replay_buffer.nb_entries, 0)
        data = agent.sample(env=env, sample_count=10, store_flag=True, in_which_status='TRAIN')
        self.assertTrue(isinstance(data, SampleData))
        self.assertEqual(agent.algo.replay_buffer.nb_entries, 10)

    def test_test(self):
        algo, local = self.create_dqn()
        env = local['env']
        env_spec = local['env_spec']
        agent, _ = self.create_agent(algo=algo, env=env,
                                     env_spec=env_spec,
                                     eps=self.create_eps(env_spec=env_spec)[0])
        self.register_global_status_when_test(agent, env)
        agent.init()
        env.reset()
        agent.test(sample_count=2)
