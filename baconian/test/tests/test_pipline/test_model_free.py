from baconian.algo.rl.model_free.dqn import DQN
from baconian.envs.gym_env import make
from baconian.core.core import EnvSpec
from baconian.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from baconian.agent.agent import Agent
from baconian.algo.rl.misc.exploration_strategy.epsilon_greedy import EpsilonGreedy
from baconian.test.tests.set_up.setup import TestWithAll


class TestModelFreePipeline(TestWithAll):
    def test_agent(self):
        dqn, locals = self.create_dqn()
        env_spec = locals['env_spec']
        env = locals['env']
        agent = Agent(env=locals['env'], algo=dqn,
                      name='agent',
                      exploration_strategy=self.create_eps(env_spec)[0],
                      env_spec=env_spec)
        model_free = self.create_model_free_pipeline(env, agent)[0]
        model_free.launch()
