from baconian.common.sampler.sample_data import TransitionData
from baconian.test.tests.set_up.setup import TestWithAll
from baconian.config.global_config import GlobalConfig


class TestDDPG(TestWithAll):
    def test_init(self):
        ddpg, locals = self.create_ddpg()
        self.assert_var_list_id_no_equal(var_list1=ddpg.actor.parameters('tf_var_list'),
                                         var_list2=ddpg.target_actor.parameters('tf_var_list'))

        self.assert_var_list_id_no_equal(var_list1=ddpg.critic.parameters('tf_var_list'),
                                         var_list2=ddpg.target_critic.parameters('tf_var_list'))

        self.assert_var_list_id_equal(var_list1=ddpg.critic.parameters('tf_var_list'),
                                      var_list2=ddpg._critic_with_actor_output.parameters('tf_var_list'))
        self.assert_var_list_id_equal(var_list1=ddpg.target_critic.parameters('tf_var_list'),
                                      var_list2=ddpg._target_critic_with_target_actor_output.parameters('tf_var_list'))

        env = locals['env']
        env_spec = locals['env_spec']
        ddpg.init()
        data = TransitionData(env_spec)
        st = env.reset()
        for i in range(100):
            ac = ddpg.predict(st)
            new_st, re, done, _ = env.step(ac)
            data.append(state=st, new_state=new_st, action=ac, reward=re, done=done)
            st = new_st
        ddpg.append_to_memory(data)
        new_ddpg, _ = self.create_ddpg(name='new_ddpg')
        new_ddpg.copy_from(ddpg)
        self.assert_var_list_equal(ddpg.actor.parameters('tf_var_list'),
                                   new_ddpg.actor.parameters('tf_var_list'))
        self.assert_var_list_equal(ddpg.critic.parameters('tf_var_list'),
                                   new_ddpg.critic.parameters('tf_var_list'))
        self.assert_var_list_equal(ddpg.target_actor.parameters('tf_var_list'),
                                   new_ddpg.target_actor.parameters('tf_var_list'))
        self.assert_var_list_equal(ddpg.target_critic.parameters('tf_var_list'),
                                   new_ddpg.target_critic.parameters('tf_var_list'))
        ddpg.save(save_path=GlobalConfig().DEFAULT_LOG_PATH + '/ddpg_test',
                  global_step=0,
                  name=ddpg.name)
        for i in range(100):
            print(ddpg.train(train_iter=10))
        self.assert_var_list_at_least_not_equal(ddpg.critic.parameters('tf_var_list'),
                                                new_ddpg.critic.parameters('tf_var_list'))
        self.assert_var_list_at_least_not_equal(ddpg.target_critic.parameters('tf_var_list'),
                                                new_ddpg.target_critic.parameters('tf_var_list'))
        self.assert_var_list_at_least_not_equal(ddpg.actor.parameters('tf_var_list'),
                                                new_ddpg.actor.parameters('tf_var_list'))

        self.assert_var_list_at_least_not_equal(ddpg.target_actor.parameters('tf_var_list'),
                                                new_ddpg.target_actor.parameters('tf_var_list'))

        ddpg.load(path_to_model=GlobalConfig().DEFAULT_LOG_PATH + '/ddpg_test',
                  model_name=ddpg.name,
                  global_step=0)

        self.assert_var_list_equal(ddpg.actor.parameters('tf_var_list'),
                                   new_ddpg.actor.parameters('tf_var_list'))
