from mobrl.test.tests.set_up.setup import TestTensorflowSetup


class TestAgent(TestTensorflowSetup):
    def test_agent(self):
        algo, local = self.create_dqn()
        env = local['env']
        env_spec = local['env_spec']
        agent, _ = self.create_agent(algo=algo, env=env,
                                     env_spec=env_spec,
                                     eps=self.create_eps(env_spec=env_spec)[0])

        agent.init()
        env.reset()
        data = agent.sample(env=env, sample_count=10, store_flag=True, in_test_flag=False)
        from mobrl.common.sampler.sample_data import SampleData
        self.assertTrue(isinstance(data, SampleData))
        self.assertEqual(agent.algo.replay_buffer.nb_entries, 10)
