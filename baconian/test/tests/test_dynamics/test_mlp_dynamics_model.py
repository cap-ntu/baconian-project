from baconian.common.sampler.sample_data import TransitionData
from baconian.test.tests.set_up.setup import TestWithAll
import numpy as np


class TestDynamicsModel(TestWithAll):

    def test_mlp_dynamics_model(self):
        mlp_dyna, local = self.create_continue_dynamics_model(name='mlp_dyna_model')
        env = local['env']
        env_spec = local['env_spec']
        env.reset()
        mlp_dyna.init()
        for i in range(100):
            mlp_dyna.step(action=np.array(env_spec.action_space.sample()),
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
        mlp_dyna_2, _ = self.create_continue_dynamics_model(name='model_2')
        mlp_dyna_2.init()
        self.assert_var_list_at_least_not_equal(var_list1=mlp_dyna.parameters('tf_var_list'),
                                                var_list2=mlp_dyna_2.parameters('tf_var_list'))

        self.assert_var_list_id_no_equal(var_list1=mlp_dyna.parameters('tf_var_list'),
                                         var_list2=mlp_dyna_2.parameters('tf_var_list'))

        mlp_dyna_2.init(source_obj=mlp_dyna)

        self.assert_var_list_equal(var_list1=mlp_dyna.parameters('tf_var_list'),
                                   var_list2=mlp_dyna_2.parameters('tf_var_list'))

        self.assert_var_list_id_no_equal(var_list1=mlp_dyna.parameters('tf_var_list'),
                                         var_list2=mlp_dyna_2.parameters('tf_var_list'))

        mlp_dyna_2.copy_from(mlp_dyna)

        self.assert_var_list_equal(var_list1=mlp_dyna.parameters('tf_var_list'),
                                   var_list2=mlp_dyna_2.parameters('tf_var_list'))

        self.assert_var_list_id_no_equal(var_list1=mlp_dyna.parameters('tf_var_list'),
                                         var_list2=mlp_dyna_2.parameters('tf_var_list'))
