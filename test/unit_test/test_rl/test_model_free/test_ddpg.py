import unittest
from src.rl.algo.model_free import DQN
from src.envs.gym_env import make
from src.envs.env_spec import EnvSpec
from src.rl.value_func.mlp_q_value import MLPQValueFunction
import tensorflow as tf
from src.tf.util import create_new_tf_session
import numpy as np
from src.rl.algo.model_free.ddpg import DDPG
from src.rl.policy.deterministic_mlp import DeterministicMLPPolicy
from src.common.sampler.sample_data import TransitionData


class TestDDPG(unittest.TestCase):
    def test_init(self):
        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
            # sess.close()
        tf.reset_default_graph()
        env = make('Swimmer-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        sess = create_new_tf_session(cuda_device=0)

        mlp_q = MLPQValueFunction(env_spec=env_spec,
                                  name_scope='mlp_q',
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
        policy = DeterministicMLPPolicy(env_spec=env_spec,
                                        name_scope='mlp_policy',
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
                                                "N_UNITS": env_spec.flat_action_dim,
                                                "TYPE": "DENSE",
                                                "W_NORMAL_STDDEV": 0.03
                                            }
                                        ],
                                        reuse=False)
        ddpg = DDPG(
            env_spec=env_spec,
            config_or_config_dict={
                "REPLAY_BUFFER_SIZE": 10000,
                "GAMMA": 0.999,
                "Q_NET_L1_NORM_SCALE": 0.01,
                "Q_NET_L2_NORM_SCALE": 0.01,
                "CRITIC_LEARNING_RATE": 0.001,
                "ACTOR_LEARNING_RATE": 0.001,
                "DECAY": 0.5,
                "ACTOR_BATCH_SIZE": 5,
                "CRITIC_BATCH_SIZE": 5,
                "CRITIC_TRAIN_ITERATION": 1,
                "ACTOR_TRAIN_ITERATION": 1,
                "critic_clip_norm": 0.001
            },
            value_func=mlp_q,
            policy=policy,
            adaptive_learning_rate=True,
            name='ddpg',
            replay_buffer=None
        )
        ddpg.init()
        data = TransitionData(env_spec)
        st = env.reset()
        for i in range(100):
            ac = ddpg.predict(st)
            new_st, re, done, _ = env.step(ac)
            data.append(state=st, new_state=new_st, action=ac, reward=re, done=done)
        ddpg.append_to_memory(data)
        for i in range(1000):
            print(ddpg.train())
