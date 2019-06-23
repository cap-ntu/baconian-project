from baconian.test.tests.set_up.setup import TestWithAll
from baconian.algo.dynamics.reward_func.reward_func import QuadraticCostFunc
from baconian.envs.gym_env import make
import numpy as np
from baconian.core.core import EnvSpec
from baconian.algo.dynamics.linear_dynamics_model import LinearDynamicsModel
from baconian.algo.policy.lqr_policy import LQRPolicy


class TestLQRPolicy(TestWithAll):
    default_id = -1

    def test_correctness(self):
        env_id = 'Pendulum-v0'
        env = make(env_id)
        n = env.observation_space.flat_dim
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        F = np.ones([env.observation_space.flat_dim,
                     env.observation_space.flat_dim + env.action_space.flat_dim]) * 0.00001
        # F[n:, n:] = 0.0001
        dyna = LinearDynamicsModel(env_spec=env_spec,
                                   state_transition_matrix=F,
                                   bias=np.zeros([env.observation_space.flat_dim]))
        C = np.ones([env.observation_space.flat_dim + env.action_space.flat_dim,
                     env.observation_space.flat_dim + env.action_space.flat_dim]) * 0.00001
        c = np.ones([env.observation_space.flat_dim + env.action_space.flat_dim])
        c[n:] = -1000
        # C[:n, :] = 0.
        # C[:, :n] = 0.
        # c[:n] = 0.0
        cost_fn = QuadraticCostFunc(C=C, c=c)

        policy = LQRPolicy(env_spec=env_spec,
                           T=5,
                           dynamics=dyna,
                           cost_fn=cost_fn)
        st = env.reset() * 0.0
        for i in range(10):
            ac = policy.forward(st)
            st = dyna.step(action=ac, state=st, allow_clip=True)
            print(cost_fn(state=st, action=ac, new_state=None))
            print(st, ac)
