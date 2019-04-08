from baconian.common.sampler.sample_data import TransitionData
from baconian.test.tests.set_up.setup import TestWithAll
import numpy as np


class TestDyna(TestWithAll):

    def test_init(self):
        ddpg, locals = self.create_ddpg()
        env_spec = locals['env_spec']
        env = locals['env']
        mlp_dyna = self.create_continuous_mlp_global_dynamics_model(env_spec=env_spec)[0]
        algo = self.create_dyna(env_spec=env_spec, model_free_algo=ddpg, dyanmics_model=mlp_dyna)[0]
        algo.init()

        st = env.reset()
        data = TransitionData(env_spec)

        for _ in range(100):
            ac = algo.predict(st)
            new_st, re, done, _ = env.step(action=ac)
            data.append(state=st,
                        new_state=new_st,
                        reward=re,
                        action=ac,
                        done=done)
        algo.append_to_memory(samples=data)
        pre_res = 10000
        for i in range(20):
            print(algo.train(batch_data=data))
            print(algo.train(batch_data=data, state='state_dynamics_training'))
            print(algo.train(batch_data=data, state='state_agent_training'))
            res = algo.test_dynamics(env=env, sample_count=100)
            self.assertLess(list(res.values())[0], pre_res)
            self.assertLess(list(res.values())[1], pre_res)
            print(res)
        algo.test()
