from mbrl.algo.rl.model_free.dqn import DQN
from mbrl.envs.gym_env import make
from mbrl.envs.env_spec import EnvSpec
from mbrl.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from mbrl.test.tests.test_setup import TestTensorflowSetup
from mbrl.core.util import get_global_arg_dict, copy_globally
import tensorflow as tf


class TestExp(TestTensorflowSetup):
    def test_init_arg_decorator(self):
        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

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
                  adaptive_learning_rate=True,
                  config_or_config_dict=dict(REPLAY_BUFFER_SIZE=1000,
                                             GAMMA=0.99,
                                             BATCH_SIZE=10,
                                             Q_NET_L1_NORM_SCALE=0.001,
                                             Q_NET_L2_NORM_SCALE=0.001,
                                             LEARNING_RATE=0.001,
                                             TRAIN_ITERATION=1,
                                             DECAY=0.5),
                  value_func=mlp_q)
        dqn.init()
        a = get_global_arg_dict()
        self.assertTrue(dqn in a)
        self.assertTrue(env_spec in a)
        self.assertTrue(mlp_q in a)
        print(a.keys())
        self.setUp()
        new_dqn = copy_globally(arg_dict=a, source_obj_list=[dqn])[0]
        new_dqn.init()

        a = get_global_arg_dict()

        self.assertTrue(new_dqn in a)
        self.assertTrue(dqn not in a)
        print(a)
        self.setUp()
        new_dqn_2 = copy_globally(arg_dict=a, source_obj_list=[new_dqn])[0]
        new_dqn_2.init()
        a = get_global_arg_dict()

        self.assertTrue(new_dqn_2 in a)
        self.assertTrue(new_dqn not in a)
        print(a)
