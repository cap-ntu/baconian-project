from mbrl.envs.gym_env import GymEnv
from mbrl.test.tests.test_setup import BaseTestCase


class TestEnv(BaseTestCase):
    def test_gym_env(self):
        a = GymEnv('Acrobot-v1')
        a.init()
        a.reset()
        a.seed(10)
        a.step(action=a.action_space.sample())

    def test_all_get_state(self):
        type_list = []
        # for id in GymEnv._all_gym_env_id:
        #     try:
        #         env = make(id)
        #         type_list.append(type(env).__name__)
        #         env.reset()
        #         st = env.get_state()
        #         self.assertTrue(env.observation_space.contains(st))
        #         assert env.observation_space.contains(st)
        #     except BaseException:
        #         print("{} is not found".format(id))
        print(set(type_list))
