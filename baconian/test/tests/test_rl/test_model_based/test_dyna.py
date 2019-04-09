from baconian.common.sampler.sample_data import TransitionData
from baconian.test.tests.set_up.setup import TestWithAll
import numpy as np
from baconian.algo.dynamics.terminal_func.terminal_func import *
from baconian.algo.dynamics.reward_func.reward_func import RandomRewardFunc


class TestDynamics(TestWithAll):

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

    def test_dynamics_as_env(self):
        env = self.create_env('Pendulum-v0')
        env_spec = self.create_env_spec(env)

        mlp_dyna = self.create_continuous_mlp_global_dynamics_model(env_spec=env_spec)[0]
        env = mlp_dyna.return_as_env()
        env.init()
        env.set_terminal_reward_func(terminal_func=FixedEpisodeLengthTerminalFunc(max_step_length=10,
                                                                                  step_count_fn=lambda: env.total_step_count_fn() - env._last_reset_point),
                                     reward_func=RandomRewardFunc())
        env.reset()
        self.assertEqual(env._last_reset_point, 0)
        for i in range(11):
            new_st, re, done, _ = env.step(action=env_spec.action_space.sample())
            self.assertEqual(env.total_step_count_fn(), i + 1)
            if done is True:
                self.assertEqual(i, 9)
                env.reset()
                self.assertEqual(env._last_reset_point, env.total_step_count_fn())
                self.assertEqual(env._last_reset_point, i + 1)
