from baconian.common.sampler.sample_data import TransitionData
import unittest
from baconian.test.tests.set_up.setup import TestTensorflowSetup


class TestMPC(TestTensorflowSetup):

    def test_init_discrete(self):
        algo, locals = self.create_mpc()
        env_spec = locals['env_spec']
        env = locals['env']
        algo.init()
        for _ in range(100):
            assert env_spec.action_space.contains(algo.predict(env_spec.obs_space.sample()))

        st = env.reset()
        data = TransitionData(env_spec)

        for _ in range(10):
            ac = algo.predict(st)
            new_st, re, done, _ = env.step(action=ac)
            data.append(state=st,
                        new_state=new_st,
                        reward=re,
                        action=ac,
                        done=done)
        print(algo.train(batch_data=data))

    def test_init_continuous(self):
        algo, locals = self.create_mpc(env_id='Pendulum-v0')
        env_spec = locals['env_spec']
        env = locals['env']
        algo.init()
        for _ in range(100):
            assert env_spec.action_space.contains(algo.predict(env_spec.obs_space.sample()))

        st = env.reset()
        data = TransitionData(env_spec)

        for _ in range(10):
            ac = algo.predict(st)
            new_st, re, done, _ = env.step(action=ac)
            data.append(state=st,
                        new_state=new_st,
                        reward=re,
                        action=ac,
                        done=done)
        print(algo.train(batch_data=data))

    def test_mpc_polymorphism(self):
        policy_func = (
            self.create_mlp_deterministic_policy, self.create_normal_dist_mlp_policy, self.create_uniform_policy,
            self.create_constant_action_policy)
        for i, func in enumerate(policy_func):
            self.setUp()
            wrap_policy(self, func=func)()
            if i < len(policy_func) - 1:
                self.tearDown()


def wrap_policy(self, func):
    def wrap_func():
        mlp_dyna, local = self.create_continue_dynamics_model(env_id='Pendulum-v0')
        env_spec = local['env_spec']
        env = local['env']
        policy = func(env_spec=env_spec)[0]
        algo, locals = self.create_mpc(env_spec=env_spec, mlp_dyna=mlp_dyna, policy=policy, env=env)
        algo.init()
        for _ in range(100):
            assert env_spec.action_space.contains(algo.predict(env_spec.obs_space.sample()))

        st = env.reset()
        data = TransitionData(env_spec)

        for _ in range(10):
            ac = algo.predict(st)
            new_st, re, done, _ = env.step(action=ac)
            data.append(state=st,
                        new_state=new_st,
                        reward=re,
                        action=ac,
                        done=done)
        print(algo.train(batch_data=data))

    return wrap_func


if __name__ == '__main__':
    unittest.main()
