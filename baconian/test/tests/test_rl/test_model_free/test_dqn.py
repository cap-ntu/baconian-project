from baconian.test.tests.set_up.setup import TestWithAll
from baconian.config.global_config import GlobalConfig


class TestDQN(TestWithAll):
    def test_init(self):
        dqn, locals = self.create_dqn()
        env = locals['env']
        env_spec = locals['env_spec']
        dqn.init()
        st = env.reset()
        from baconian.common.sampler.sample_data import TransitionData
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

        dqn.save(save_path=GlobalConfig.DEFAULT_LOG_PATH + '/dqn_test',
                 global_step=0,
                 name=dqn.name)

        for i in range(20):
            # print(dqn.train(batch_data=a, train_iter=10, sess=None, update_target=True))
            print(dqn.train(batch_data=None, train_iter=10, sess=None, update_target=True))

        self.assert_var_list_at_least_not_equal(dqn.q_value_func.parameters('tf_var_list'),
                                                new_dqn.q_value_func.parameters('tf_var_list'))

        self.assert_var_list_at_least_not_equal(dqn.target_q_value_func.parameters('tf_var_list'),
                                                new_dqn.target_q_value_func.parameters('tf_var_list'))

        dqn.load(path_to_model=GlobalConfig.DEFAULT_LOG_PATH + '/dqn_test',
                 model_name=dqn.name,
                 global_step=0)

        self.assert_var_list_equal(dqn.q_value_func.parameters('tf_var_list'),
                                   new_dqn.q_value_func.parameters('tf_var_list'))
        self.assert_var_list_equal(dqn.target_q_value_func.parameters('tf_var_list'),
                                   new_dqn.target_q_value_func.parameters('tf_var_list'))
