# import numpy as np
from scipy.linalg import inv

from baconian.algo.policy.policy import DeterministicPolicy
from baconian.core.parameters import Parameters
from baconian.core.core import EnvSpec
from baconian.algo.dynamics.linear_dynamics_model import LinearDynamicsModel
import autograd.numpy as np
from baconian.common.special import *
from baconian.algo.dynamics.reward_func.reward_func import CostFunc, QuadraticCostFunc


class LQR(object):

    def __init__(self, env_spec: EnvSpec, T, dyna_model: LinearDynamicsModel, cost_fn: QuadraticCostFunc):

        self.env_spec = env_spec
        self.T = T
        self.dyn_model = dyna_model
        self.cost_fn = cost_fn
        self.control_low = self.env_spec.action_space.low
        self.control_high = self.env_spec.action_space.high
        self.K, self.k, self.std = None, None, None
        self.C = np.repeat(np.expand_dims(self.cost_fn.C, axis=0), [self.T], axis=0)
        self.F = np.repeat(np.expand_dims(self.dyn_model.F, axis=0), [self.T], axis=0)
        self.c = np.repeat(np.expand_dims(self.cost_fn.c, axis=0), [self.T], axis=0)
        self.f = np.repeat(np.expand_dims(self.dyn_model.f, axis=0), [self.T], axis=0)

    def differentiate(self):

        "get gradient values using finite difference"

        C = np.repeat(np.expand_dims(self.cost_fn.C, axis=0), [self.T, 1, 1])
        F = np.repeat(np.expand_dims(self.dyn_model.F, axis=0), [self.T, 1, 1])
        c = np.repeat(np.expand_dims(self.cost_fn.c, axis=0), (self.T, 1))
        f = np.repeat(np.expand_dims(self.dyn_model.f, axis=0), (self.T, 1))
        return C, F, c, f

    def backward(self, x_seq, u_seq):

        "initialize F_t, C_t, f_t, c_t, V_t, v_t"
        # todo C, F, c, f can be analytical set here
        C, F, c, f = self.C, self.F, self.c, self.f

        n = x_seq[0].shape[0]

        "initialize V_t1 and v_t1"

        c_x = c[-1][:n]
        c_u = c[-1][n:]

        C_xx = C[-1][:n, :n]
        C_xu = C[-1][:n, n:]
        C_ux = C[-1][n:, :n]
        C_uu = C[-1][n:, n:]

        # C_uu = C_uu + self.mu * np.eye(C_uu.shape[0])

        K = np.zeros((self.T + 1, u_seq[0].shape[0], x_seq[0].shape[0]))
        k = np.zeros((self.T + 1, u_seq[0].shape[0]))

        V = np.zeros((self.T + 1, x_seq[0].shape[0], x_seq[0].shape[0]))
        v = np.zeros((self.T + 1, x_seq[0].shape[0]))

        # K[-1] = -np.dot(inv(C_uu), C_ux)
        # k[-1] = -np.dot(inv(C_uu), c_u)

        # V[-1] = C_xx + np.dot(C_xu, K[-1]) + np.dot(K[-1].T, C_ux) + np.dot(np.dot(K[-1].T, C_uu), K[-1])
        # v[-1] = c_x + np.dot(C_xu, k[-1]) + np.dot(K[-1].T, c_u) + np.dot(np.dot(K[-1].T, C_uu), k[-1])
        V[-1] = C_xx
        v[-1] = c_x

        "initialize Q_t1 and q_t1"

        Q = list(np.zeros((self.T)))
        q = list(np.zeros((self.T)))

        self.std = []
        "loop till horizon"

        t = self.T - 1
        while t >= 0:
            Q[t] = C[t] + np.dot(np.dot(F[t].T, V[t + 1]), F[t])
            q[t] = c[t] + np.dot(F[t].T, v[t + 1]) + np.dot(np.dot(F[t].T, V[t + 1]), f[t])

            "differentiate Q to get Q_uu, Q_xx, Q_ux, Q_u, Q_x"

            q_x = q[t][:n]
            q_u = q[t][n:]

            Q_xx = Q[t][:n, :n]
            Q_xu = Q[t][:n, n:]
            Q_ux = Q[t][n:, :n]
            Q_uu = Q[t][n:, n:]

            "update K, k, V, v"
            # print("q_uu", Q_uu)

            K[t] = -np.dot(inv(Q_uu), Q_ux)
            k[t] = -np.dot(inv(Q_uu), q_u)

            V[t] = Q_xx + np.dot(Q_xu, K[t]) + np.dot(K[t].T, Q_ux) + np.dot(np.dot(K[t].T, Q_uu), K[t])
            v[t] = q_x + np.dot(Q_xu, k[t]) + np.dot(K[t].T, q_u) + np.dot(np.dot(K[t].T, Q_uu), k[t])

            self.std.append(inv(Q_uu))
            t -= 1

        self.K = K
        self.k = k

    def get_action_one_step(self, state, t):

        mean = np.dot(self.K[t], state) + self.k[t]
        # todo remove random here
        if np.isclose(0.0, self.std[t]).all() is False:
            return np.clip(np.random.normal(mean, self.std[t]), self.control_low, self.control_high)
        else:
            return np.clip(mean, self.control_low, self.control_high)


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
