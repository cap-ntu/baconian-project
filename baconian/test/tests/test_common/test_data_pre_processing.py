from baconian.envs.gym_env import make
from baconian.core.core import EnvSpec
from baconian.test.tests.set_up.setup import BaseTestCase
from baconian.common.data_pre_processing import *
import numpy as np


class TestDataPreProcessing(BaseTestCase):
    def test_min_max(self):
        for env in (make('Pendulum-v0'), make('Acrobot-v1'), make('RoboschoolAnt-v1')):
            for sample_space in (env.observation_space, env.action_space):
                sample_fn = sample_space.sample
                dims = sample_space.flat_dim
                try:
                    print("test {} with sample {} dims {}".format(env, sample_fn, dims))
                    # test batch scaler
                    min_max = BatchMinMaxScaler(dims=dims)
                    data_list = []
                    for i in range(100):
                        data_list.append(sample_fn())
                    data = min_max.process(np.array(data_list))
                    self.assertTrue(np.greater_equal(np.ones(dims),
                                                     data).all())
                    self.assertTrue(np.less_equal(np.zeros(dims),
                                                  data).all())
                    # test batch scaler with given range
                    min_max = BatchMinMaxScaler(dims=dims,
                                                desired_range=(np.ones(dims) * -1.0,
                                                               np.ones(dims) * 5.0))
                    data_list = []
                    for i in range(100):
                        data_list.append(sample_fn())
                    data = min_max.process(np.array(data_list))
                    self.assertTrue(np.greater_equal(np.ones(dims) * 5.0,
                                                     data).all())
                    self.assertTrue(np.less_equal(np.ones(dims) * -1.0,
                                                  data).all())
                    self.assertEqual(np.max(data), 5.0)
                    self.assertEqual(np.min(data), -1.0)
                    data = min_max.inverse_process(data)
                    self.assertTrue(np.isclose(data, np.array(data_list)).all())

                    # test batch scaler with given range and given initial data
                    data_list = []
                    for i in range(100):
                        data_list.append(sample_fn())

                    min_max = RunningMinMaxScaler(dims=dims,
                                                  desired_range=(np.ones(dims) * -1.0,
                                                                 np.ones(dims) * 5.0),
                                                  init_data=np.array(data_list))

                    data = min_max.process(np.array(data_list))
                    self.assertTrue(np.greater_equal(np.ones(dims) * 5.0,
                                                     data).all())
                    self.assertTrue(np.less_equal(np.ones(dims) * -1.0,
                                                  data).all())
                    self.assertEqual(np.max(data), 5.0)
                    self.assertEqual(np.min(data), -1.0)

                    # test batch scaler with given range and given initial min and max
                    data_list = []
                    for i in range(100):
                        data_list.append(sample_fn())

                    min_max = RunningMinMaxScaler(dims=dims,
                                                  desired_range=(np.ones(dims) * -1.0,
                                                                 np.ones(dims) * 5.0),
                                                  init_min=np.min(np.array(data_list), axis=0),
                                                  init_max=np.max(np.array(data_list), axis=0))

                    data = min_max.process(np.array(data_list))
                    self.assertTrue(np.greater_equal(np.ones(dims) * 5.0,
                                                     data).all())
                    self.assertTrue(np.less_equal(np.ones(dims) * -1.0,
                                                  data).all())
                    self.assertEqual(np.max(data), 5.0)
                    self.assertEqual(np.min(data), -1.0)

                    # test update function by a larger range of data
                    pre_min = np.min(np.array(data_list), axis=0)
                    pre_max = np.max(np.array(data_list), axis=0)
                    data_list = np.array(data_list) * 2.0
                    min_max.update_scaler(data_list)
                    self.assertTrue(np.equal(pre_min * 2.0, min_max._min).all())
                    self.assertTrue(np.equal(pre_max * 2.0, min_max._max).all())
                except ShapeNotCompatibleError as e:
                    from baconian.common.spaces import Box
                    if isinstance(sample_space, Box):
                        raise ValueError
                    else:
                        pass

    def test_standard_scaler(self):
        for env in (make('Pendulum-v0'), make('Acrobot-v1'), make('RoboschoolAnt-v1')):
            for sample_space in (env.observation_space, env.action_space):
                sample_fn = sample_space.sample
                dims = sample_space.flat_dim
                try:
                    # test batch standard scaler
                    standard_scaler = BatchStandardScaler(dims=dims)
                    data_list = []
                    for i in range(100):
                        data_list.append(sample_fn())
                    data = standard_scaler.process(np.array(data_list))
                    self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())
                    # TODO a theoretical bound should be given
                    # self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())
                    data = standard_scaler.inverse_process(data)
                    self.assertTrue(np.isclose(data, np.array(data_list)).all())

                    # test running standard scaler
                    standard_scaler = RunningStandardScaler(dims=dims)
                    data_list = []
                    for i in range(100):
                        data_list.append(sample_fn())
                    standard_scaler.update_scaler(np.array(data_list))
                    self.assertEqual(standard_scaler._data_count, 100)
                    data = standard_scaler.process(np.array(data_list))
                    self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())

                    # TODO a theoretical bound should be given
                    # self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())
                    # test update function
                    new_data_list = []
                    for i in range(100):
                        new_data_list.append(sample_fn())
                    standard_scaler.update_scaler(np.array(new_data_list))
                    self.assertEqual(standard_scaler._data_count, 200)

                    data_list += new_data_list
                    data = standard_scaler.process(np.array(data_list))
                    self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())

                    # TODO a theoretical bound should be given
                    # self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())

                    # test running scaler with given data
                    data_list = []
                    for i in range(100):
                        data_list.append(sample_fn())
                    standard_scaler = RunningStandardScaler(dims=dims,
                                                            init_data=np.array(data_list))

                    self.assertEqual(standard_scaler._data_count, 100)
                    data = standard_scaler.process(np.array(data_list))
                    self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())
                    # TODO a theoretical bound should be given
                    # self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())
                    # test update of running scaler with given data
                    new_data_list = []
                    for i in range(100):
                        new_data_list.append(sample_fn())
                    standard_scaler.update_scaler(np.array(new_data_list))
                    self.assertEqual(standard_scaler._data_count, 200)

                    data_list += new_data_list
                    data = standard_scaler.process(np.array(data_list))
                    self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())

                    # TODO a theoretical bound should be given
                    # self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())

                    # test running scaler with given initial mean, var.
                    data_list = []
                    for i in range(100):
                        data_list.append(sample_fn())
                    standard_scaler = RunningStandardScaler(dims=dims,
                                                            init_mean=np.mean(data_list, axis=0),
                                                            init_var=np.var(data_list, axis=0),
                                                            init_mean_var_data_count=100)

                    self.assertEqual(standard_scaler._data_count, 100)
                    data = standard_scaler.process(np.array(data_list))
                    self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())
                    # TODO a theoretical bound should be given
                    # self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())

                    new_data_list = []
                    for i in range(100):
                        new_data_list.append(sample_fn())
                    standard_scaler.update_scaler(np.array(new_data_list))
                    self.assertEqual(standard_scaler._data_count, 200)

                    data_list += new_data_list
                    data = standard_scaler.process(np.array(data_list))
                    self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())

                    # TODO a theoretical bound should be given
                    # self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())
                except ShapeNotCompatibleError as e:
                    from baconian.common.spaces import Box
                    if isinstance(sample_space, Box):
                        raise ValueError
                    else:
                        pass
