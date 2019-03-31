# from baconian.test.tests.set_up.setup import TestWithAll
# import numpy as np
# from baconian.algo.dynamics.linear_dynamics_model import LinearDynamicsModel
# from baconian.common.sampler.sample_data import TransitionData
#
#
# class TestDynamicsModel(TestWithAll):
#
#     def test_dynamics_model(self):
#         gp, local = self.create_gp_dynamics()
#         env_spec = local['env_spec']
#         env = local['env']
#         policy, _ = self.create_uniform_policy(env_spec=env_spec)
#         data = TransitionData(env_spec=env_spec)
#         st = env.reset()
#         for i in range(100):
#             ac = policy.forward(st)
#             new_st, re, _, _ = env.step(action=ac)
#             data.append(state=st, new_state=new_st, reward=re, done=False, action=ac)
#             st = new_st
#         gp.init(batch_data=data)
