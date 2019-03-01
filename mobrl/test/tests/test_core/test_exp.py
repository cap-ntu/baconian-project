from mobrl.agent.agent import Agent
from mobrl.test.tests.set_up.setup import BaseTestCase
from mobrl.core.experiment import exp_runner


class TestExperiment(BaseTestCase):
    def test_experiment(self):
        def func():
            dqn, locals = self.create_dqn()
            env_spec = locals['env_spec']
            env = locals['env']
            agent = Agent(env=locals['env'], algo=dqn,
                          name='agent',
                          exploration_strategy=self.create_eps(env_spec)[0],
                          env_spec=env_spec)
            model_free = self.create_model_free_pipeline(env, agent)[0]
            exp = self.create_exp(pipeline=model_free, name='model_fre')
            exp.run()

        exp_runner(func, auto_choose_gpu_flag=False, gpu_id=0)
