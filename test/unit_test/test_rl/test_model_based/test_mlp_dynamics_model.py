from src.rl.algo.model_based.models.dynamics_model import DynamicsModel
from src.rl.algo.model_based.models.mlp_dynamics_model import ContinuousMLPDynamicsModel
import unittest
import unittest
from src.rl.algo.model_free import DQN
from src.envs.gym_env import make
from src.envs.env_spec import EnvSpec
from src.rl.value_func.mlp_q_value import MLPQValueFunction
import tensorflow as tf
from src.tf.util import create_new_tf_session
import numpy as np
from src.common.sampler.sample_data import TransitionData


class TestDynamicsModel(unittest.TestCase):

    def test_mlp_dynamics_model(self):

        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
            # sess.close()
        tf.reset_default_graph()
        env = make('Acrobot-v1')
        env.reset()
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        sess = create_new_tf_session(cuda_device=0)
        mlp_dyna = ContinuousMLPDynamicsModel(
            env_spec=env_spec,
            name_scope='mlp_dyna',
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
                    "N_UNITS": 6,
                    "TYPE": "DENSE",
                    "W_NORMAL_STDDEV": 0.03
                }
            ])
        mlp_dyna.init()
        mlp_dyna.step(action=env_spec.action_space.sample(),
                      state=env_spec.obs_space.sample())
        data = TransitionData(env_spec)
        st = env.get_state()
        for i in range(10):
            ac = env_spec.action_space.sample()
            new_st, re, done, info = env.step(action=ac)
            data.append(state=st,
                        action=ac,
                        new_state=new_st,
                        done=done,
                        reward=re)
            st = new_st
        print(mlp_dyna.train(batch_data=data, train_iter=10))

        mlp_dyna_2 = ContinuousMLPDynamicsModel(
            env_spec=env_spec,
            name_scope='mlp_dyna2',
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
                    "N_UNITS": 6,
                    "TYPE": "DENSE",
                    "W_NORMAL_STDDEV": 0.03
                }
            ])
        mlp_dyna_2.init(source_obj=mlp_dyna)
        mlp_dyna_2.copy(mlp_dyna)
