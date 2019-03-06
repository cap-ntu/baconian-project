from baconian.envs.gym_env import GymEnv
from baconian.test.tests.set_up.setup import BaseTestCase
from gym import make


class TestEnv(BaseTestCase):
    def test_gym_env(self):
        a = GymEnv('Acrobot-v1')
        a.init()
        a.reset()
        a.seed(10)
        a.step(action=a.action_space.sample())

    def test_all_get_state(self):
        type_list = []
        for id in GymEnv._all_gym_env_id:
            try:
                env = make(id)
                type_list.append(type(env).__name__)
                env.reset()
                st = env.get_state()
                self.assertTrue(env.observation_space.contains(st))
                assert env.observation_space.contains(st)
            except BaseException:
                print("{} is not found".format(id))
        print(set(type_list))
