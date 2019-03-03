# from baconian.envs.gym_env import make
# from baconian.core.core import EnvSpec
# from baconian.algo.rl.value_func.mlp_q_value import MLPQValueFunction
# from baconian.agent.agent import Agent
# from baconian.algo.rl.misc.epsilon_greedy import EpsilonGreedy
# from baconian.core.pipelines.model_based_pipeline import ModelBasedPipeline
# from baconian.algo.rl.model_based.models.mlp_dynamics_model import ContinuousMLPGlobalDynamicsModel
# from baconian.algo.rl.model_based.sample_with_model import SampleWithDynamics
# from baconian.test.tests.set_up.setup import TestWithAll
# from baconian.algo.rl.model_free.dqn import DQN
#
#
# class TestModelFreePipeline(TestWithAll):
#     def test_agent(self):
#         dqn, locals = self.create_dqn()
#         env_spec = locals['env_spec']
#         env = locals['env']
#         mlp_dyna = self.create_continuous_mlp_global_dynamics_model(env_spec=env_spec)[0]
#
#         algo = self.create_sample_with_model_algo(env_spec=env_spec,
#                                                   model_free_algo=dqn,
#                                                   dyanmics_model=mlp_dyna)[0]
#         agent = self.create_agent(algo=algo, env=locals['env'], env_spec=locals['env_spec'],
#                                   eps=self.create_eps(env_spec)[0])
#         pipeline = ModelBasedPipeline(agent=agent, env=env,
#                                       config_or_config_dict=dict(TEST_SAMPLES_COUNT=100,
#                                                                  TRAIN_SAMPLES_COUNT=100,
#                                                                  TOTAL_SAMPLES_COUNT=1000))
#         pipeline.launch()
