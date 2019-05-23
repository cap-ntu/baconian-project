from baconian.test.tests.set_up.setup import TestTensorflowSetup
from baconian.core.util import get_global_arg_dict, copy_globally


class TestCopyGlobally(TestTensorflowSetup):
    def test_init_arg_decorator(self):
        dqn, local = self.create_dqn()
        env_spec = local['env_spec']
        mlp_q = local['mlp_q']
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
        self.assertNotEqual(id(new_dqn), id(dqn))

        print(a)
        self.setUp()
        new_dqn_2 = copy_globally(arg_dict=a, source_obj_list=[new_dqn])[0]
        new_dqn_2.init()
        a = get_global_arg_dict()

        self.assertTrue(new_dqn_2 in a)
        self.assertTrue(new_dqn not in a)
        self.assertNotEqual(id(new_dqn_2), id(dqn))
        self.assertNotEqual(id(new_dqn_2), id(new_dqn))
        print(a)
