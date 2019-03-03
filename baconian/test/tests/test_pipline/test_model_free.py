# from baconian.agent.agent import Agent
# from baconian.test.tests.set_up.setup import TestWithAll
#
#
# class TestModelFreePipeline(TestWithAll):
#     def test_agent(self):
#         dqn, locals = self.create_dqn()
#         env_spec = locals['env_spec']
#         env = locals['env']
#         agent = self.create_agent(algo=dqn, env=locals['env'], env_spec=locals['env_spec'], eps=self.create_eps(env_spec)[0])
#
#         model_free = self.create_model_free_pipeline(env, agent)[0]
#         model_free.launch()
