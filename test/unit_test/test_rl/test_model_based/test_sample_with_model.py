from src.rl.algo.model_based.sample_with_model import SampleWithDynamics
import unittest
from src.rl.algo.model_free import DQN
from src.envs.gym_env import make
from src.envs.env_spec import EnvSpec
from src.rl.value_func.mlp_q_value import MLPQValueFunction
import tensorflow as tf
from src.tf.util import create_new_tf_session
import numpy as np
from src.rl.algo.model_based.models.mlp_dynamics_model import ContinuousMLPDynamicsModel
from src.common.sampler.sample_data import TransitionData


class TestSampleWithDynamics(unittest.TestCase):

    def test_init_discrete(self):
        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
            # sess.close()
        tf.reset_default_graph()
        env = make('Acrobot-v1')
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
        dqn = DQN(env_spec=env_spec,
                  config_or_config_dict=dict(REPLAY_BUFFER_SIZE=1000,
                                             GAMMA=0.99,
                                             BATCH_SIZE=10,
                                             Q_NET_L1_NORM_SCALE=0.001,
                                             Q_NET_L2_NORM_SCALE=0.001,
                                             LEARNING_RATE=0.001,
                                             TRAIN_ITERATION=1,
                                             DECAY=0.5),
                  value_func=mlp_q)
        mlp_dyna = ContinuousMLPDynamicsModel(
            env_spec=env_spec,
            name_scope='mlp_dyna',
            input_norm=False,
            output_norm=False,
            output_low=env_spec.obs_space.low,
            output_high=env_spec.obs_space.high,
            l1_norm_scale=1.0,
            l2_norm_scale=1.0,
            learning_rate=0.01,
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
                    "N_UNITS": env_spec.flat_obs_dim,
                    "TYPE": "DENSE",
                    "W_NORMAL_STDDEV": 0.03
                }
            ])
        algo = SampleWithDynamics(env_spec=env_spec,
                                  model_free_algo=dqn,
                                  dynamics_model=mlp_dyna,
                                  config_or_config_dict=dict(
                                      dynamics_model_train_iter=1,
                                      model_free_algo_train_iter=1
                                  ))
        algo.init()
        for _ in range(100):
            assert env_spec.action_space.contains(algo.predict(env_spec.obs_space.sample()))

        st = env.reset()
        data = TransitionData(env_spec)

        for _ in range(100):
            ac = algo.predict(st)
            new_st, re, done, _ = env.step(action=ac)
            data.append(state=st,
                        new_state=new_st,
                        reward=re,
                        action=ac,
                        done=done)
        algo.append_to_memory(samples=data)
        print(algo.train(batch_data=data))


if __name__ == '__main__':
    unittest.main()
