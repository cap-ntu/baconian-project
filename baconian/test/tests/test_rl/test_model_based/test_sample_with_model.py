from baconian.common.sampler.sample_data import TransitionData
from baconian.test.tests.set_up.setup import TestWithAll


class TestSampleWithDynamics(TestWithAll):

    def test_init_discrete(self):
        dqn, locals = self.create_dqn()
        env_spec = locals['env_spec']
        env = locals['env']
        mlp_dyna = self.create_continuous_mlp_global_dynamics_model(env_spec=env_spec)[0]
        algo = self.create_sample_with_model_algo(env_spec=env_spec, model_free_algo=dqn, dyanmics_model=mlp_dyna)[0]
        algo.init()
        for _ in range(100):
            assert env_spec.action_space.contains(algo.predict(env_spec.obs_space.sample()))

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
        for i in range(100):
            print(algo.train(batch_data=data))
