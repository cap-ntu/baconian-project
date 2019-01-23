import unittest
from src.rl.algo.model_free import DQN
from gym import make
from src.envs.env_spec import EnvSpec
from src.rl.value_func.mlp_q_value import MLPQValueFunction
import tensorflow as tf
from src.tf.util import create_new_tf_session
import numpy as np
from src.rl.exploration_strategy.epsilon_greedy import EpsilonGreedy


class TestExplorationStrategy(unittest.TestCase):
    def test_eps_greedy(self):
        tf.reset_default_graph()

        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        sess = create_new_tf_session(cuda_device=0)

        mlp_q = MLPQValueFunction(env_spec=env_spec,
                                  name_scope='mlp_q',
                                  input_norm=False,
                                  output_norm=False,
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
        dqn = DQN(env=env,
                  config_or_config_dict=dict(REPLAY_BUFFER_SIZE=1000,
                                             GAMMA=0.99,
                                             BATCH_SIZE=10,
                                             Q_NET_L1_NORM_SCALE=0.001,
                                             Q_NET_L2_NORM_SCALE=0.001,
                                             LEARNING_RATE=0.001,
                                             DECAY=0.5),
                  value_func=mlp_q)
        dqn.init()
        eps = EpsilonGreedy(action_space=dqn.env_spec.action_space,
                            init_random_prob=0.5,
                            decay_type=None)
        st = env.reset()
        for i in range(100):
            # ac = dqn.predict(obs=st, sess=sess, batch_flag=False)
            ac = eps.predict(obs=st, sess=sess, batch_flag=False, algo=dqn)
            st_new, re, done, _ = env.step(action=ac)
            print(ac)
