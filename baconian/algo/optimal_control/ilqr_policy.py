from baconian.algo.rl.policy.policy import DeterministicPolicy
from baconian.core.parameters import Parameters
from baconian.core.core import EnvSpec
from baconian.algo.optimal_control.ilqr import iLQR as iLQR_algo
from baconian.algo.dynamics.dynamics_model import DynamicsModel
import autograd.numpy as np
from baconian.common.special import *
from baconian.algo.dynamics.reward_func.reward_func import CostFunc

"""
the gradient is computed approximated instead of analytically
"""


class iLQRPolicy(DeterministicPolicy):

    @typechecked
    def __init__(self, env_spec: EnvSpec, T: int, delta: float, iteration: int, cost_fn: CostFunc,
                 dynamics: DynamicsModel):
        param = Parameters(parameters=dict(T=T, delta=delta, iteration=iteration))
        super().__init__(env_spec, param)
        self.dynamics = dynamics
        self.U_hat = None
        self.X_hat = None
        self.iLqr_instance = iLQR_algo(env_spec=env_spec,
                                       delta=self.parameters('delta'),
                                       T=self.parameters('T'),
                                       dyn_model=dynamics,
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
        return iLQRPolicy(env_spec=self.env_spec,
                          T=self.parameters('T'),
                          delta=self.parameters('delta'),
                          iteration=self.parameters('iteration'),
                          cost_fn=self.iLqr_instance.cost_fn,
                          dynamics=self.dynamics.make_copy())

    def init(self, source_obj=None):
        self.parameters.init()
        if source_obj:
            self.copy_from(obj=source_obj)

    def get_status(self):
        return super().get_status()

    def _forward(self, obs, step: None):
        if not step:
            self.U_hat = np.reshape([np.zeros(self.action_space.sample().shape) for _ in range(self.T - 1)],
                                    (self.T - 1, self.action_space.shape[0]))

            self.X_hat = []
            self.X_hat.append(obs)
            x = obs
            for i in range(self.T - 1):
                next_obs = self.dynamics.step(action=self.U_hat[i, :], state=x)
                self.X_hat.append(next_obs)
                x = next_obs
            self.X_hat = np.array(self.X_hat)
        for i in range(self.parameters('iteration')):
            self.iLqr_instance.backward(self.X_hat, self.U_hat)
            x = obs
            U = np.zeros(self.U_hat.shape)
            X = np.zeros(self.X_hat.shape)

            for t in range(self.T - 1):
                u = self.iLqr_instance.get_action_one_step(x, t, self.X_hat[t], self.U_hat[t])
                X[t] = x
                U[t] = u
                x = self.dynamics.step(state=x, action=u)

            X[-1] = x
            self.X_hat = X
            self.U_hat = U
        return self.U_hat[0]

    @property
    def T(self):
        return self.parameters('T')
