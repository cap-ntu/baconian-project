from baconian.test.tests.set_up.setup import TestTensorflowSetup
from baconian.core.util import get_global_arg_dict


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

        a = get_global_arg_dict()
