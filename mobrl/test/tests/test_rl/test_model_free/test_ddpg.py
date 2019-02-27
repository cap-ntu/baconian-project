from mobrl.common.sampler.sample_data import TransitionData
from mobrl.test.tests.set_up.setup import TestWithAll
from mobrl.config.global_config import GlobalConfig


class TestDDPG(TestWithAll):
    def test_init(self):
        ddpg, locals = self.create_ddpg()
        env = locals['env']
        env_spec = locals['env_spec']
        ddpg.init()
        data = TransitionData(env_spec)
        st = env.reset()
        for i in range(100):
            ac = ddpg.predict(st)
            new_st, re, done, _ = env.step(ac)
            data.append(state=st, new_state=new_st, action=ac, reward=re, done=done)
        ddpg.append_to_memory(data)
        new_ddpg, _ = self.create_ddpg(name='new_ddpg')
        new_ddpg.copy_from(ddpg)
        self.assert_var_list_equal(ddpg.actor.parameters('tf_var_list'),
                                   new_ddpg.actor.parameters('tf_var_list'))
        for i in range(2):
            ddpg.save(save_path=GlobalConfig.DEFAULT_LOG_PATH + '/ddpg_test',
                      global_step=i,
                      name=ddpg.name)
            print(ddpg.train())
        self.assert_var_list_at_least_not_equal(ddpg.actor.parameters('tf_var_list'),
                                                new_ddpg.actor.parameters('tf_var_list'))

        ddpg.load(path_to_model=GlobalConfig.DEFAULT_LOG_PATH + '/ddpg_test',
                  model_name=ddpg.name,
                  global_step=0)

        self.assert_var_list_equal(ddpg.actor.parameters('tf_var_list'),
                                   new_ddpg.actor.parameters('tf_var_list'))
