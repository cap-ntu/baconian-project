from baconian.common.sampler.sample_data import TransitionData
from baconian.test.tests.set_up.setup import TestWithAll
import numpy as np
from baconian.algo.rl.model_based.models.linear_dynamics_model import LinearDynamicsModel


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
