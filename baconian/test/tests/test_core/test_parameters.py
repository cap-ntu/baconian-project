from baconian.config.global_config import GlobalConfig
from baconian.test.tests.set_up.setup import TestWithLogSet
import numpy as np
from baconian.core.parameters import Parameters
from baconian.common.schedules import LinearScheduler, PiecewiseScheduler, PeriodicalEventSchedule

x = 0


class TestParam(TestWithLogSet):
    def test_basic(self):
        a, locals = self.create_parameters()
        a.save(save_path=GlobalConfig().DEFAULT_LOG_PATH + '/param_path',
               name=a.name,
               global_step=0)
        or_val = a._source_config.config_dict['var1']
        or_param = a('param3').copy()
        a._source_config.config_dict['var1'] = 100
        a._parameters['param3'] = 1000
        self.assertNotEqual(a._source_config.config_dict['var1'], or_val)
        self.assertFalse(np.equal(a._parameters['param3'], or_param).all())
        a.load(load_path=GlobalConfig().DEFAULT_LOG_PATH + '/param_path',
               name=a.name,
               global_step=0
               )
        self.assertEqual(a._source_config.config_dict['var1'], or_val)
        self.assertTrue(np.equal(a._parameters['param3'], or_param).all())
        b, _ = self.create_parameters()
        b.copy_from(a)

        for key in a._source_config.required_key_dict.keys():
            if isinstance(a[key], np.ndarray):
                self.assertTrue(np.equal(a[key], b[key]).all())
            else:
                self.assertEqual(id(a[key]), id(b[key]))
                self.assertEqual(id(a(key)), id(b(key)))
        for key in a._parameters.keys():
            if isinstance(a[key], np.ndarray):
                self.assertTrue(np.equal(a[key], b[key]).all())
            else:
                self.assertEqual(a[key], b[key])
                self.assertEqual(a(key), b(key))

    def test_scheduler_param(self):
        def func():
            global x
            return x

        parameters = dict(param1='aaaa',
                          param2=1.0,
                          param4=1.0,
                          param3=np.random.random([4, 2]))
        source_config, _ = self.create_dict_config()
        a = Parameters(parameters=parameters,
                       source_config=source_config,
                       name='test_params',
                       to_scheduler_param_tuple=(dict(param_key='param2',
                                                      scheduler=LinearScheduler(t_fn=func,
                                                                                schedule_timesteps=10,
                                                                                final_p=0.0)),
                                                 dict(param_key='param4',
                                                      scheduler=PiecewiseScheduler(t_fn=func,
                                                                                   endpoints=(
                                                                                      (2, 0.5), (8, 0.2), (10, 0.0)),
                                                                                   outside_value=0.0,
                                                                                   ))))
        a.init()
        for i in range(20):
            global x
            if x < 10:
                self.assertEqual(a('param2'), 1.0 - x * (1.0 - 0.0) / 10)
            else:
                self.assertEqual(a('param2'), 0.0)
            if x == 2:
                self.assertEqual(a('param4'), 0.5)
            if x == 8:
                self.assertEqual(a('param4'), 0.2)
            if x >= 10:
                self.assertEqual(a('param4'), 0.0)
            x += 1
        b, _ = self.create_parameters()
        b.copy_from(a)
        for key in a._source_config.required_key_dict.keys():
            if isinstance(a[key], np.ndarray):
                self.assertTrue(np.equal(a[key], b[key]).all())
            else:
                self.assertEqual(id(a[key]), id(b[key]))
                self.assertEqual(id(a(key)), id(b(key)))
        for key in a._parameters.keys():
            if isinstance(a[key], np.ndarray):
                self.assertTrue(np.equal(a[key], b[key]).all())
            else:
                self.assertEqual(a[key], b[key])
                self.assertEqual(a(key), b(key))
        self.assertEqual(a.to_scheduler_param_list.__len__(), b.to_scheduler_param_list.__len__())
        for a_val, b_val in zip(a.to_scheduler_param_list, b.to_scheduler_param_list):
            self.assertEqual(a_val['param_key'], b_val['param_key'])
            self.assertEqual(a_val['scheduler'].value(), b_val['scheduler'].value())

    def test_event_schedule(self):
        def func():
            global x
            return x

        a = PeriodicalEventSchedule(t_fn=func,
                                    trigger_every_step=5,
                                    after_t=10)
        for i in range(100):
            if i % 5 != 0 or i < 10:
                self.assertFalse(a.value())
            else:
                self.assertTrue(a.value())
            global x
            x += 1
