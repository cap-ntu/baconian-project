from baconian.test.tests.set_up.setup import TestWithAll
from baconian.algo.dynamics.reward_func.reward_func import CostFunc
from baconian.envs.gym_env import make
import numpy as np
from baconian.core.core import EnvSpec
from baconian.algo.dynamics.dynamics_model import GlobalDynamicsModel
from baconian.algo.policy.ilqr_policy import iLQRPolicy
from baconian.algo.dynamics.dynamics_model import DynamicsEnvWrapper
from baconian.algo.dynamics.terminal_func.terminal_func import RandomTerminalFunc


class DebugDynamics(GlobalDynamicsModel):
    flag = 0.5
    st = None

    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        return state + 0.0001 * action
        # self.flag *= -1.0
        # return np.ones_like(self.env_spec.obs_space.sample()) * self.flag
        # return self.env_spec.obs_space.sample()


class DebuggingCostFunc(CostFunc):
    def __call__(self, state=None, action=None, new_state=None, **kwargs) -> float:
        # return float(np.sum(action * action) + np.sum(state * state))
        return float(np.sum(action + action * action))


class TestiLQRPolicy(TestWithAll):
    def test_correctness(self):
        env_id = 'Pendulum-v0'

        env = make(env_id)
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        dyna = DebugDynamics(env_spec=env_spec)
        dyna = DynamicsEnvWrapper(dynamics=dyna)
        dyna.set_terminal_reward_func(terminal_func=RandomTerminalFunc(),
                                      reward_func=DebuggingCostFunc())
        policy = iLQRPolicy(env_spec=env_spec,
                            T=10,
                            delta=0.05,
                            iteration=2,
                            dynamics=dyna,
                            dynamics_model_train_iter=10,
                            cost_fn=DebuggingCostFunc())
        st = env.reset()
        dyna.st = np.zeros_like(st)
        for i in range(10):
            ac = policy.forward(st)
            st, _, _, _ = env.step(st)
            # st = dyna.step(action=ac, state=st)
            print("analytical optimal action -0.5, cost -0.25")
            print('state: {}, action: {}, cost {}'.format(st, ac, policy.iLqr_instance.cost_fn(state=st, action=ac,
                                                                                               new_state=None)))
