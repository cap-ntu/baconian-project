from baconian.test.tests.set_up.setup import TestWithAll
from baconian.core.status import StatusCollector


class TestStatus(TestWithAll):

    def test_status_collector(self):
        a = StatusCollector()

        algo, local = self.create_dqn()
        env = local['env']
        env_spec = local['env_spec']
        agent, _ = self.create_agent(algo=algo, env=env,
                                     env_spec=env_spec,
                                     eps=self.create_eps(env_spec=env_spec)[0])
        self.register_global_status_when_test(env=env, agent=agent)

        agent.init()

        a.register_info_key_status(obj=agent, info_key='predict_counter', under_status='TRAIN',
                                   return_name='train_counter')
        a.register_info_key_status(obj=agent, info_key='predict_counter', under_status='TEST',
                                   return_name='test_counter')
        env.reset()
        agent.sample(env=env, sample_count=10, store_flag=True, in_which_status='TRAIN')
        agent.sample(env=env, sample_count=10, store_flag=True, in_which_status='TEST')
        agent.sample(env=env, sample_count=10, store_flag=True, in_which_status='TRAIN')

        res = a()
        self.assertTrue(len(res) == 2)

        self.assertTrue('train_counter' in res)
        self.assertTrue('test_counter' in res)

        self.assertTrue(res['test_counter'] == 10)
        self.assertTrue(res['train_counter'] == 20)


class TestStatusWithDQN(TestWithAll):
    def test_with_dqn(self):
        dqn, local = self.create_dqn()
        env = local['env']
        env_spec = local['env_spec']
        dqn.init()
        st = env.reset()
        from baconian.common.sampler.sample_data import TransitionData
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
