from mbrl.test.tests.testSetup import TestTensorflowSetup

from mbrl.rl.algo.model_free import DQN
from mbrl.envs.gym_env import make
from mbrl.envs.env_spec import EnvSpec
from mbrl.rl.value_func.mlp_q_value import MLPQValueFunction
from mbrl.agent.agent import Agent
from mbrl.rl.misc.exploration_strategy.epsilon_greedy import EpsilonGreedy


class TestAgent(TestTensorflowSetup):
    def test_agent(self):
        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        mlp_q = MLPQValueFunction(env_spec=env_spec,
                                  name_scope='mlp_q',
                                  output_low=None,
                                  output_high=None,
                                  mlp_config=[
                                      {
                                          "ACT": "RELU",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "1",
                                          "N_UNITS": 16,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      },
                                      {
                                          "ACT": "LINEAR",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "OUPTUT",
                                          "N_UNITS": 1,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      }
                                  ])
        dqn = DQN(env_spec=env_spec,
                  config_or_config_dict=dict(REPLAY_BUFFER_SIZE=1000,
                                             GAMMA=0.99,
                                             BATCH_SIZE=10,
                                             Q_NET_L1_NORM_SCALE=0.001,
                                             Q_NET_L2_NORM_SCALE=0.001,
                                             LEARNING_RATE=0.001,
                                             TRAIN_ITERATION=10,
                                             DECAY=0.5),
                  value_func=mlp_q)
        agent = Agent(env=env, env_spec=env_spec,
                      algo=dqn,
                      exploration_strategy=EpsilonGreedy(action_space=dqn.env_spec.action_space,
                                                         init_random_prob=0.5,
                                                         decay_type=None))
        agent.init()
        env.reset()
        data = agent.sample(env=env, sample_count=10, store_flag=True, in_test_flag=False)
        from mbrl.common.sampler.sample_data import SampleData
        self.assertTrue(isinstance(data, SampleData))
        self.assertEqual(agent.algo.replay_buffer.nb_entries, 10)
