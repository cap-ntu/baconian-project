from baconian.algo.rl.policy.policy import DeterministicPolicy
from baconian.core.parameters import Parameters
from baconian.core.core import EnvSpec
from baconian.algo.dynamics.linear_dynamics_model import LinearDynamicsModel
import autograd.numpy as np
from baconian.common.special import *
from baconian.algo.dynamics.reward_func.reward_func import CostFunc
from baconian.algo.optimal_control.lqr import LQR

"""
the gradient is computed approximated instead of analytically
"""


class LQRPolicy(DeterministicPolicy):

    @typechecked
    def __init__(self, env_spec: EnvSpec, T: int, cost_fn: CostFunc,
                 dynamics: LinearDynamicsModel):
        param = Parameters(parameters=dict(T=T))
        super().__init__(env_spec, param)
        self.dynamics = dynamics
        self.Lqr_instance = LQR(env_spec=env_spec,
                                T=self.parameters('T'),
                                dyna_model=dynamics,
                                cost_fn=cost_fn)

    def forward(self, obs, **kwargs):
        obs = make_batch(obs, original_shape=self.env_spec.obs_shape).tolist()
        action = []
        if 'step' in kwargs:
            step = kwargs['step']
        else:
            step = None
        for obs_i in obs:
            action_i = self._forward(obs_i, step=step)
            action.append(action_i)
        return np.array(action)

    def copy_from(self, obj) -> bool:
        super().copy_from(obj)
        self.parameters.copy_from(obj.parameters)
        return True

    def make_copy(self, *args, **kwargs):
        return LQRPolicy(env_spec=self.env_spec,
                         T=self.parameters('T'),
                         cost_fn=self.Lqr_instance.cost_fn,
                         dynamics=self.dynamics.make_copy())

    def init(self, source_obj=None):
        self.parameters.init()
        if source_obj:
            self.copy_from(obj=source_obj)

    def get_status(self):
        return super().get_status()

    def _forward(self, obs, step: None):
        self.Lqr_instance.backward(np.array([obs]), np.array([self.action_space.sample()]))
        x = obs
        u = self.Lqr_instance.get_action_one_step(x, t=step if step else 0)
        return u

    @property
    def T(self):
        return self.parameters('T')
