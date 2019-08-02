from baconian.test.tests.set_up.setup import BaseTestCase
from baconian.common.noise import *
from baconian.common.schedules import *

t = 0


def get_t():
    global t
    return t


class TestNoise(BaseTestCase):
    def test_all_noise(self):
        action_w = LinearScheduler(t_fn=get_t,
                                   schedule_timesteps=100,
                                   final_p=1.0,
                                   initial_p=0.0)
        noise_w = LinearScheduler(t_fn=get_t,
                                  final_p=0.0,
                                  schedule_timesteps=100,
                                  initial_p=1.0)

        noise_wrapper = AgentActionNoiseWrapper(noise=OUNoise(),
                                                action_weight_scheduler=action_w,
                                                noise_weight_scheduler=noise_w)
        for i in range(101):
            print('action w {}, noise w {}'.format(noise_wrapper.action_weight_scheduler.value(),
                                                   noise_wrapper.noise_weight_scheduler.value()))
            print(noise_wrapper(action=1.0))
            if i == 100:
                self.assertEqual(noise_wrapper(action=1.0), 1.0)
            global t
            t += 1
