from baconian.common.sampler.sample_data import TransitionData, TrajectoryData
from baconian.test.tests.set_up.setup import TestWithAll
from baconian.config.global_config import GlobalConfig


class TestPPO(TestWithAll):
    def test_init(self):
        ppo, locals = self.create_ppo()
        env = locals['env']
        env_spec = locals['env_spec']
        ppo.init()

        new_ppo, _ = self.create_ppo(name='new_ppo')
        new_ppo.copy_from(ppo)

        self.assert_var_list_id_no_equal(ppo.value_func.parameters('tf_var_list'),
                                         new_ppo.value_func.parameters('tf_var_list'))
        self.assert_var_list_id_no_equal(ppo.policy.parameters('tf_var_list'),
                                         new_ppo.policy.parameters('tf_var_list'))

        self.assert_var_list_equal(ppo.value_func.parameters('tf_var_list'),
                                   new_ppo.value_func.parameters('tf_var_list'))
        self.assert_var_list_equal(ppo.policy.parameters('tf_var_list'),
                                   new_ppo.policy.parameters('tf_var_list'))

        data = TransitionData(env_spec)
        st = env.reset()
        for i in range(100):
            ac = ppo.predict(st)
            assert ac.shape[0] == 1
            self.assertTrue(env_spec.action_space.contains(ac[0]))
            new_st, re, done, _ = env.step(ac)
            if i % 9 == 0 and i > 0:
                done = True
            else:
                done = False
            data.append(state=st, new_state=new_st, action=ac, reward=re, done=done)
        traj = TrajectoryData(env_spec=env_spec)
        traj.append(data)
        ppo.append_to_memory(traj)

        ppo.save(save_path=GlobalConfig().DEFAULT_LOG_PATH + '/ppo_test',
                 global_step=0,
                 name=ppo.name)
        for i in range(5):
            ppo.append_to_memory(traj)
            res = ppo.train()

            print('value_func_loss {}, policy_average_loss: {}'.format(res['value_func_loss'],
                                                                       res['policy_average_loss']))
            traj_data = TrajectoryData(env_spec=env_spec)
            traj_data.append(data)
            res = ppo.train(trajectory_data=traj_data,
                            train_iter=5,
                            sess=self.sess)

            print('value_func_loss {}, policy_average_loss: {}'.format(res['value_func_loss'],
                                                                       res['policy_average_loss']))

        self.assert_var_list_at_least_not_equal(ppo.value_func.parameters('tf_var_list'),
                                                new_ppo.value_func.parameters('tf_var_list'))
        self.assert_var_list_at_least_not_equal(ppo.policy.parameters('tf_var_list'),
                                                new_ppo.policy.parameters('tf_var_list'))

        ppo.load(path_to_model=GlobalConfig().DEFAULT_LOG_PATH + '/ppo_test',
                 model_name=ppo.name,
                 global_step=0)

        self.assert_var_list_equal(ppo.value_func.parameters('tf_var_list'),
                                   new_ppo.value_func.parameters('tf_var_list'))
        self.assert_var_list_equal(ppo.policy.parameters('tf_var_list'),
                                   new_ppo.policy.parameters('tf_var_list'))
