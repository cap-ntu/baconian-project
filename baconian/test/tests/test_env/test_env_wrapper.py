from baconian.envs.gym_env import GymEnv, make
from baconian.test.tests.set_up.setup import TestWithLogSet
from baconian.envs.env_wrapper import StepObservationWrapper


class TestEnvWrapper(TestWithLogSet):
    def test_obs_wrapper(self):
        env = make('Pendulum-v0')
        env = StepObservationWrapper(env=env)
        env.reset()
        for i in range(10):
            obs, _, _, _ = env.step(action=env.action_space.sample())
            self.assertEqual(obs[-1], i + 1)
        obs = env.reset()
        self.assertEqual(obs[-1], 0)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertTrue(env.action_space.contains(env.action_space.sample()))
        self.assertTrue(env.observation_space.contains(env.reset()))
        env.get_state()
