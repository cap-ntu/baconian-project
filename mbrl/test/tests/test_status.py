from mbrl.core.status import Status, StatusWithSingleInfo
from mbrl.test.tests.test_setup import TestTensorflowSetup, BaseTestCase
from mbrl.test.tests.test_setup import BaseTestCase
from mbrl.common.util.logger import Logger, global_logger
from mbrl.common.util.recorder import record_return_decorator
import numpy as np
from mbrl.core.basic import Basic
from mbrl.test.tests.test_setup import TestTensorflowSetup
from mbrl.algo.rl.model_free.dqn import DQN
from mbrl.envs.gym_env import make
from mbrl.envs.env_spec import EnvSpec
from mbrl.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from mbrl.test.tests.test_setup import TestTensorflowSetup


class TestStatus(BaseTestCase):

    def test_status(self):
        pass

    def test_info_status(self):
        pass


class TestStatusWithDQN(TestTensorflowSetup):
    def test_with_dqn(self):
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
        st = env.reset()
        from mbrl.common.sampler.sample_data import TransitionData
        a = TransitionData(env_spec)
        res = []
        for i in range(100):
            ac = dqn.predict(obs=st, sess=self.sess, batch_flag=False)
            st_new, re, done, _ = env.step(action=ac)
            a.append(state=st, new_state=st_new, action=ac, done=done, reward=re)
            dqn.append_to_memory(a)
        res.append(dqn.train(batch_data=a, train_iter=10, sess=None, update_target=True)['average_loss'])
        res.append(dqn.train(batch_data=None, train_iter=10, sess=None, update_target=True)['average_loss'])
        print(dqn._status())
        print(dqn._status._info_dict_with_sub_info)
