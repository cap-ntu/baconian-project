from baconian.test.tests.set_up.setup import TestWithAll
import numpy as np
from baconian.algo.dynamics.linear_dynamics_model import LinearDynamicsModel, LinearRegressionDynamicsModel
from baconian.common.data_pre_processing import RunningStandardScaler


class TestDynamicsModel(TestWithAll):

    def test_dynamics_model(self):
        real_env = self.create_env('Pendulum-v0')
        x = real_env.observation_space.flat_dim
        u = real_env.action_space.flat_dim
        a = LinearDynamicsModel(env_spec=real_env.env_spec,
                                state_transition_matrix=np.ones((x,
                                                                 x + u)) * 0.01,
                                bias=np.ones(x) * 0.02)
        new_state = a.step(action=np.ones_like(real_env.action_space.sample()),
                           state=np.ones_like(real_env.observation_space.sample()))
        print('new state', new_state)
        true_new = np.ones([x]) * (x + u) * 0.01 + np.ones([x]) * 0.02
        print('true state', true_new)
        self.assertTrue(np.equal(true_new, new_state).all())

    def test_linear_regression_model(self):
        real_env = self.create_env('Pendulum-v0')
        real_env.init()
        x = real_env.observation_space.flat_dim
        u = real_env.action_space.flat_dim
        a = LinearRegressionDynamicsModel(env_spec=real_env.env_spec,
                                          state_input_scaler=RunningStandardScaler(
                                              dims=real_env.observation_space.flat_dim),
                                          action_input_scaler=RunningStandardScaler(
                                              dims=real_env.action_space.flat_dim),
                                          state_output_scaler=RunningStandardScaler(
                                              dims=real_env.observation_space.flat_dim))
        data = self.sample_transition(env=real_env, count=100)
        a.train(batch_data=data)
        gen = data.return_generator(shuffle_flag=True)
        predict = []
        for state, _, action, _, _ in gen:
            predict.append(a.step(state=state, action=action))
        print(np.linalg.norm(np.array(predict) - data.new_state_set, ord=1))
        print(np.linalg.norm(np.array(predict) - data.new_state_set, ord=2))
