from baconian.envs.gym_env import make
from baconian.core.core import EnvSpec
from baconian.test.tests.set_up.setup import BaseTestCase
from baconian.common.data_pre_processing import *
import numpy as np


class TestDataPreProcessing(BaseTestCase):
    def test_min_max(self):
        env = make('Pendulum-v0')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        # test batch scaler
        min_max = BatchMinMaxScaler(dims=env_spec.flat_obs_dim)
        data_list = []
        for i in range(100):
            data_list.append(env.observation_space.sample())
        data = min_max.process(np.array(data_list))
        self.assertTrue(np.greater_equal(np.ones(env_spec.flat_obs_dim),
                                         data).all())
        self.assertTrue(np.less_equal(np.zeros(env_spec.flat_obs_dim),
                                      data).all())
        # test batch scaler with given range
        min_max = BatchMinMaxScaler(dims=env_spec.flat_obs_dim,
                                    desired_range=(np.ones(env_spec.flat_obs_dim) * -1.0,
                                                   np.ones(env_spec.flat_obs_dim) * 5.0))
        data_list = []
        for i in range(100):
            data_list.append(env.observation_space.sample())
        data = min_max.process(np.array(data_list))
        self.assertTrue(np.greater_equal(np.ones(env_spec.flat_obs_dim) * 5.0,
                                         data).all())
        self.assertTrue(np.less_equal(np.ones(env_spec.flat_obs_dim) * -1.0,
                                      data).all())
        self.assertEqual(np.max(data), 5.0)
        self.assertEqual(np.min(data), -1.0)

        # test batch scaler with given range and given initial data
        data_list = []
        for i in range(100):
            data_list.append(env.observation_space.sample())

        min_max = RunningMinMaxScaler(dims=env_spec.flat_obs_dim,
                                      desired_range=(np.ones(env_spec.flat_obs_dim) * -1.0,
                                                     np.ones(env_spec.flat_obs_dim) * 5.0),
                                      init_data=np.array(data_list))

        data = min_max.process(np.array(data_list))
        self.assertTrue(np.greater_equal(np.ones(env_spec.flat_obs_dim) * 5.0,
                                         data).all())
        self.assertTrue(np.less_equal(np.ones(env_spec.flat_obs_dim) * -1.0,
                                      data).all())
        self.assertEqual(np.max(data), 5.0)
        self.assertEqual(np.min(data), -1.0)

        # test batch scaler with given range and given initial min and max
        data_list = []
        for i in range(100):
            data_list.append(env.observation_space.sample())

        min_max = RunningMinMaxScaler(dims=env_spec.flat_obs_dim,
                                      desired_range=(np.ones(env_spec.flat_obs_dim) * -1.0,
                                                     np.ones(env_spec.flat_obs_dim) * 5.0),
                                      init_min=np.min(np.array(data_list), axis=0),
                                      init_max=np.max(np.array(data_list), axis=0))

        data = min_max.process(np.array(data_list))
        self.assertTrue(np.greater_equal(np.ones(env_spec.flat_obs_dim) * 5.0,
                                         data).all())
        self.assertTrue(np.less_equal(np.ones(env_spec.flat_obs_dim) * -1.0,
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

    def test_standard_scaler(self):
        env = make('Pendulum-v0')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        # test batch standard scaler
        standard_scaler = BatchStandardScaler(dims=env_spec.flat_obs_dim)
        data_list = []
        for i in range(100):
            data_list.append(env.observation_space.sample())
        data = standard_scaler.process(np.array(data_list))
        self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())

        # TODO a theoretical bound should be given
        self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())

        # test running standard scaler
        standard_scaler = RunningStandardScaler(dims=env_spec.flat_obs_dim)
        data_list = []
        for i in range(100):
            data_list.append(env.observation_space.sample())
        standard_scaler.update_scaler(np.array(data_list))
        self.assertEqual(standard_scaler._data_count, 100)
        data = standard_scaler.process(np.array(data_list))
        self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())

        # TODO a theoretical bound should be given
        self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())
        # test update function
        new_data_list = []
        for i in range(100):
            new_data_list.append(env.observation_space.sample())
        standard_scaler.update_scaler(np.array(new_data_list))
        self.assertEqual(standard_scaler._data_count, 200)

        data_list += new_data_list
        data = standard_scaler.process(np.array(data_list))
        self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())

        # TODO a theoretical bound should be given
        self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())

        # test running scaler with given data
        data_list = []
        for i in range(100):
            data_list.append(env.observation_space.sample())
        standard_scaler = RunningStandardScaler(dims=env_spec.flat_obs_dim,
                                                init_data=np.array(data_list))

        self.assertEqual(standard_scaler._data_count, 100)
        data = standard_scaler.process(np.array(data_list))
        self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())
        # TODO a theoretical bound should be given
        self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())
        # test update of running scaler with given data
        new_data_list = []
        for i in range(100):
            new_data_list.append(env.observation_space.sample())
        standard_scaler.update_scaler(np.array(new_data_list))
        self.assertEqual(standard_scaler._data_count, 200)

        data_list += new_data_list
        data = standard_scaler.process(np.array(data_list))
        self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())

        # TODO a theoretical bound should be given
        self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())

        # test running scaler with given initial mean, var.
        data_list = []
        for i in range(100):
            data_list.append(env.observation_space.sample())
        standard_scaler = RunningStandardScaler(dims=env_spec.flat_obs_dim,
                                                init_mean=np.mean(data_list, axis=0),
                                                init_var=np.var(data_list, axis=0),
                                                init_mean_var_data_count=100)

        self.assertEqual(standard_scaler._data_count, 100)
        data = standard_scaler.process(np.array(data_list))
        self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())
        # TODO a theoretical bound should be given
        self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())

        new_data_list = []
        for i in range(100):
            new_data_list.append(env.observation_space.sample())
        standard_scaler.update_scaler(np.array(new_data_list))
        self.assertEqual(standard_scaler._data_count, 200)

        data_list += new_data_list
        data = standard_scaler.process(np.array(data_list))
        self.assertTrue(np.isclose(np.mean(data, axis=0), 0.0).all())

        # TODO a theoretical bound should be given
        self.assertTrue(np.isclose(np.var(data, axis=0), 1.0, atol=0.04).all())
