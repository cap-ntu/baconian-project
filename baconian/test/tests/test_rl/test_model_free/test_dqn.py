from baconian.test.tests.set_up.setup import TestWithAll
from baconian.config.global_config import GlobalConfig

from baconian.algo.dqn import DQN
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.common.sampler.sample_data import TransitionData
import numpy as np


class TestDQN(TestWithAll):
    def test_init(self):
        dqn, locals = self.create_dqn()
        env = locals['env']
        env_spec = locals['env_spec']
        dqn.init()
        st = env.reset()
        a = TransitionData(env_spec)
        for i in range(100):
            ac = dqn.predict(obs=st, sess=self.sess, batch_flag=False)
            st_new, re, done, _ = env.step(action=ac)
            a.append(state=st, new_state=st_new, action=ac, done=done, reward=re)
            st = st_new
            dqn.append_to_memory(a)
        new_dqn, _ = self.create_dqn(name='new_dqn')
        new_dqn.copy_from(dqn)
        self.assert_var_list_id_no_equal(dqn.q_value_func.parameters('tf_var_list'),
                                         new_dqn.q_value_func.parameters('tf_var_list'))
        self.assert_var_list_id_no_equal(dqn.target_q_value_func.parameters('tf_var_list'),
                                         new_dqn.target_q_value_func.parameters('tf_var_list'))

        self.assert_var_list_equal(dqn.q_value_func.parameters('tf_var_list'),
                                   new_dqn.q_value_func.parameters('tf_var_list'))
        self.assert_var_list_equal(dqn.target_q_value_func.parameters('tf_var_list'),
                                   new_dqn.target_q_value_func.parameters('tf_var_list'))

        dqn.save(save_path=GlobalConfig().DEFAULT_LOG_PATH + '/dqn_test',
                 global_step=0,
                 name=dqn.name)

        for i in range(10):
            print(dqn.train(batch_data=a, train_iter=10, sess=None, update_target=True))
            print(dqn.train(batch_data=None, train_iter=10, sess=None, update_target=True))

        self.assert_var_list_at_least_not_equal(dqn.q_value_func.parameters('tf_var_list'),
                                                new_dqn.q_value_func.parameters('tf_var_list'))

        self.assert_var_list_at_least_not_equal(dqn.target_q_value_func.parameters('tf_var_list'),
                                                new_dqn.target_q_value_func.parameters('tf_var_list'))

        dqn.load(path_to_model=GlobalConfig().DEFAULT_LOG_PATH + '/dqn_test',
                 model_name=dqn.name,
                 global_step=0)

        self.assert_var_list_equal(dqn.q_value_func.parameters('tf_var_list'),
                                   new_dqn.q_value_func.parameters('tf_var_list'))
        self.assert_var_list_equal(dqn.target_q_value_func.parameters('tf_var_list'),
                                   new_dqn.target_q_value_func.parameters('tf_var_list'))
        for i in range(10):
            self.sess.run(dqn.update_target_q_value_func_op,
                          feed_dict=dqn.parameters.return_tf_parameter_feed_dict())
            var1 = self.sess.run(dqn.q_value_func.parameters('tf_var_list'))
            var2 = self.sess.run(dqn.target_q_value_func.parameters('tf_var_list'))
            import numpy as np
            total_diff = 0.0
            for v1, v2 in zip(var1, var2):
                total_diff += np.mean(np.abs(np.array(v1) - np.array(v2)))
            print('update target, difference mean', total_diff)

    def test_l1_l2_norm(self):
        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        name = 'dqn'

        mlp_q = MLPQValueFunction(env_spec=env_spec,
                                  name_scope=name + '_mlp',
                                  name=name + '_mlp',
                                  mlp_config=[
                                      {
                                          "ACT": "RELU",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "1",
                                          "N_UNITS": 16,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03,
                                          "L1_NORM": 1000.0,
                                          "L2_NORM": 1000.0
                                      },
                                      {
                                          "ACT": "LINEAR",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "OUPTUT",
                                          "N_UNITS": 1,
                                          "L1_NORM": 1000.0,
                                          "L2_NORM": 1000.0,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      }
                                  ])
        dqn = DQN(env_spec=env_spec,
                  config_or_config_dict=dict(REPLAY_BUFFER_SIZE=1000,
                                             GAMMA=0.99,
                                             BATCH_SIZE=10,
                                             LEARNING_RATE=0.01,
                                             TRAIN_ITERATION=1,
                                             DECAY=0.5),
                  name=name,
                  value_func=mlp_q)
        dqn2, _ = self.create_dqn(name='dqn_2')
        a = TransitionData(env_spec)
        st = env.reset()
        dqn.init()
        dqn2.init()
        for i in range(100):
            ac = dqn.predict(obs=st, sess=self.sess, batch_flag=False)
            st_new, re, done, _ = env.step(action=ac)
            a.append(state=st, new_state=st_new, action=ac, done=done, reward=re)
            st = st_new
            dqn.append_to_memory(a)
        for i in range(20):
            print('dqn1 loss: ', dqn.train(batch_data=a, train_iter=10, sess=None, update_target=True))
            print('dqn2 loss: ', dqn2.train(batch_data=a, train_iter=10, sess=None, update_target=True))
        var_list = self.sess.run(dqn.q_value_func.parameters('tf_var_list'))
        print(var_list)
        var_list2 = self.sess.run(dqn2.q_value_func.parameters('tf_var_list'))
        print(var_list2)
        for var, var2 in zip(var_list, var_list2):
            diff = np.abs(var2) - np.abs(var)
            self.assertTrue(np.greater(np.mean(diff), 0.0).all())
