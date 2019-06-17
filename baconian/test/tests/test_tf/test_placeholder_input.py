from baconian.test.tests.set_up.setup import TestWithAll
from baconian.config.global_config import GlobalConfig
import glob


class TestPlaceholderInput(TestWithAll):
    def test_tf_param(self):
        a, _ = self.create_ph('test')
        for i in range(5):
            a.save(save_path=GlobalConfig().DEFAULT_LOG_PATH + '/test_placehoder_input',
                   global_step=i,
                   name='a')
        file = glob.glob(GlobalConfig().DEFAULT_LOG_PATH + '/test_placehoder_input/a*.meta')
        self.assertTrue(len(file) == 5)
        b, _ = self.create_ph('b')
        b.copy_from(obj=a)
        self.assert_var_list_equal(a.parameters('tf_var_list'),
                                   b.parameters('tf_var_list'))

        a.parameters.init()
        self.assert_var_list_at_least_not_equal(a.parameters('tf_var_list'),
                                                b.parameters('tf_var_list'))

        a.load(path_to_model=GlobalConfig().DEFAULT_LOG_PATH + '/test_placehoder_input',
               global_step=4,
               model_name='a')

        self.assert_var_list_equal(a.parameters('tf_var_list'),
                                   b.parameters('tf_var_list'))

    def test_save_load_with_dqn(self):
        dqn, locals = self.create_dqn()
        dqn.init()
        for i in range(5):
            dqn.save(save_path=GlobalConfig().DEFAULT_LOG_PATH + '/test_placehoder_input', global_step=i, name='dqn')
        file = glob.glob(GlobalConfig().DEFAULT_LOG_PATH + '/test_placehoder_input/dqn*.meta')
        self.assertTrue(len(file) == 5)
        dqn2, _ = self.create_dqn(name='dqn_2')
        dqn2.copy_from(dqn)

        self.assert_var_list_equal(dqn.parameters('tf_var_list'), dqn2.parameters('tf_var_list'))
        self.assert_var_list_equal(dqn.q_value_func.parameters('tf_var_list'),
                                   dqn2.q_value_func.parameters('tf_var_list'))
        self.assert_var_list_equal(dqn.target_q_value_func.parameters('tf_var_list'),
                                   dqn2.target_q_value_func.parameters('tf_var_list'))

        dqn.init()
        self.assert_var_list_at_least_not_equal(dqn.q_value_func.parameters('tf_var_list'),
                                                dqn2.q_value_func.parameters('tf_var_list'))
        self.assert_var_list_at_least_not_equal(dqn.target_q_value_func.parameters('tf_var_list'),
                                                dqn2.target_q_value_func.parameters('tf_var_list'))
        dqn.load(path_to_model=GlobalConfig().DEFAULT_LOG_PATH + '/test_placehoder_input', global_step=4,
                 model_name='dqn')

        self.assert_var_list_equal(dqn.parameters('tf_var_list'), dqn2.parameters('tf_var_list'))
        self.assert_var_list_equal(dqn.q_value_func.parameters('tf_var_list'),
                                   dqn2.q_value_func.parameters('tf_var_list'))
        self.assert_var_list_equal(dqn.target_q_value_func.parameters('tf_var_list'),
                                   dqn2.target_q_value_func.parameters('tf_var_list'))
