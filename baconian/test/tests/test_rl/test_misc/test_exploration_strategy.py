from baconian.algo.rl.misc.epsilon_greedy import EpsilonGreedy
from baconian.test.tests.set_up.setup import TestWithAll
from baconian.common.util.schedules import LinearSchedule

x = 0


class TestExplorationStrategy(TestWithAll):
    def test_eps_greedy(self):
        dqn, locals = self.create_dqn()
        dqn.init()
        env = locals['env']
        eps = EpsilonGreedy(action_space=dqn.env_spec.action_space,
                            init_random_prob=0.5)
        st = env.reset()
        for i in range(100):
            ac = eps.predict(obs=st, sess=self.sess, batch_flag=False, algo=dqn)
            st_new, re, done, _ = env.step(action=ac)
            self.assertTrue(env.action_space.contains(ac))

    def test_eps_with_scheduler(self):
        dqn, locals = self.create_dqn()
        env = locals['env']

        def func():
            global x
            return x

        dqn.init()
        eps = EpsilonGreedy(action_space=dqn.env_spec.action_space,
                            init_random_prob=1.0)
        eps.parameters.set_scheduler(param_key='init_random_prob',
                                     scheduler=LinearSchedule(initial_p=1.0, t_fn=func, schedule_timesteps=10,
                                                              final_p=0.0))
        st = env.reset()
        for i in range(10):
            global x
            ac = eps.predict(obs=st, sess=self.sess, batch_flag=False, algo=dqn)
            st_new, re, done, _ = env.step(action=ac)
            print()
            self.assertAlmostEqual(eps.parameters('init_random_prob'), 1.0 - (1.0 - 0.0) / 10 * x)
            x += 1
