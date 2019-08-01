# import numpy as np
from scipy.linalg import inv
from scipy.optimize import approx_fprime

from baconian.algo.policy.policy import DeterministicPolicy
from baconian.core.parameters import Parameters
from baconian.core.core import EnvSpec
import autograd.numpy as np
from baconian.common.special import *
from baconian.algo.dynamics.reward_func.reward_func import CostFunc
from baconian.algo.rl_algo import ModelBasedAlgo
from baconian.common.logging import record_return_decorator
from baconian.core.status import register_counter_info_to_status_decorator
from baconian.common.sampler.sample_data import TransitionData

from baconian.algo.dynamics.dynamics_model import DynamicsEnvWrapper

"""
the gradient is computed approximated instead of analytically
"""


class iLQR(object):

    def __init__(self, env_spec: EnvSpec, delta, T, dyn_model: DynamicsEnvWrapper, cost_fn: CostFunc):

        self.env_spec = env_spec
        self.min_factor = 2
        self.factor = self.min_factor
        self.min_mu = 1e-6
        self.mu = self.min_mu
        self.delta = delta
        self.T = T
        self.dyn_model = dyn_model
        self.cost_fn = cost_fn
        self.control_low = self.env_spec.action_space.low
        self.control_high = self.env_spec.action_space.high
        self.K, self.k, self.std = None, None, None

    def increase(self, mu):
        self.factor = np.maximum(self.factor, self.factor * self.min_factor)
        self.mu = np.maximum(self.min_mu, self.mu * self.factor)

    def decrease(self, mu):
        self.factor = np.minimum(1 / self.min_factor, self.factor / self.min_factor)
        if self.mu * self.factor > self.min_mu:
            self.mu = self.mu * self.factor
        else:
            self.mu = 0

    def simulate_step(self, x):

        xu = [x[:self.env_spec.obs_space.shape[0]], x[self.env_spec.obs_space.shape[0]:]]

        next_x = self.dyn_model.step(state=xu[0], action=xu[1], allow_clip=True)
        # next_x = self.env_spec.obs_space.sample()
        "get cost"
        cost = self.cost_fn(state=xu[0], action=xu[1], new_state=next_x)

        return next_x, cost

    def simulate_next_state(self, x, i):
        return self.simulate_step(x)[0][i]

    def simulate_cost(self, x):
        return self.simulate_step(x)[1]

    def approx_fdoubleprime(self, x, i):
        return approx_fprime(x, self.simulate_cost, self.delta)[i]

    def finite_difference(self, x, u):

        "calling finite difference for delta perturbation"
        xu = np.concatenate((x, u))

        F = np.zeros((x.shape[0], xu.shape[0]))

        for i in range(x.shape[0]):
            F[i, :] = approx_fprime(xu, self.simulate_next_state, self.delta, i)

        c = approx_fprime(xu, self.simulate_cost, self.delta)

        C = np.zeros((len(xu), len(xu)))

        for i in range(xu.shape[0]):
            C[i, :] = approx_fprime(xu, self.approx_fdoubleprime, self.delta, i)

        f = np.zeros((len(x)))

        return C, F, c, f

    def differentiate(self, x_seq, u_seq):

        "get gradient values using finite difference"

        C, F, c, f = [], [], [], []

        for t in range(self.T - 1):
            Ct, Ft, ct, ft = self.finite_difference(x_seq[t], u_seq[t])

            C.append(Ct)
            F.append(Ft)
            c.append(ct)
            f.append(ft)

        "TODO : C, F, c, f for time step T are different. Why ?"

        u = np.zeros((u_seq[0].shape))

        Ct, Ft, ct, ft = self.finite_difference(x_seq[-1], u)

        C.append(Ct)
        F.append(Ft)
        c.append(ct)
        f.append(ft)

        return C, F, c, f

    def backward(self, x_seq, u_seq):

        "initialize F_t, C_t, f_t, c_t, V_t, v_t"
        C, F, c, f = self.differentiate(x_seq, u_seq)

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
            # for t in range(self.T-1, -1, -1):
            "update Q"
            # todo remove mu here
            # Q[t] = C[t] + np.dot(np.dot(F[t].T, V[t + 1] + self.mu * np.eye(V[t + 1].shape[0])),
            #                      F[t])  # + 0.01 * np.eye(V[t+1].shape[0])),

            Q[t] = C[t] + np.dot(np.dot(F[t].T, V[t + 1]), F[t])
            q[t] = c[t] + np.dot(F[t].T, v[t + 1]) + np.dot(np.dot(F[t].T, V[t + 1]), f[t])

            "differentiate Q to get Q_uu, Q_xx, Q_ux, Q_u, Q_x"

            q_x = q[t][:n]
            q_u = q[t][n:]

            Q_xx = Q[t][:n, :n]
            Q_xu = Q[t][:n, n:]
            Q_ux = Q[t][n:, :n]
            Q_uu = Q[t][n:, n:]

            # Q_uu = Q_uu + 100 * np.eye(Q_uu.shape[0])
            # try:
            #     np.linalg.cholesky(Q_uu)
            # except:
            #     print(self.mu)
            #     self.increase(self.mu)
            #     continue

            "update K, k, V, v"
            # print("q_uu", Q_uu)

            K[t] = -np.dot(inv(Q_uu), Q_ux)
            k[t] = -np.dot(inv(Q_uu), q_u)

            V[t] = Q_xx + np.dot(Q_xu, K[t]) + np.dot(K[t].T, Q_ux) + np.dot(np.dot(K[t].T, Q_uu), K[t])
            v[t] = q_x + np.dot(Q_xu, k[t]) + np.dot(K[t].T, q_u) + np.dot(np.dot(K[t].T, Q_uu), k[t])

            self.std.append(inv(Q_uu))
            # self.decrease(self.mu)
            t -= 1

        self.K = K
        self.k = k

    def get_action_one_step(self, state, t, x, u):

        "TODO : Add delta U's to given action array"
        # print("Q_uu ", self.std[t])
        mean = np.dot(self.K[t], (state - x)) + self.k[t] + u
        # return np.clip(mean, self.control_low, self.control_high)
        # todo remove random here
        if np.isclose(0.0, self.std[t]).all() is False:
            return np.clip(np.random.normal(mean, self.std[t]), self.control_low, self.control_high)
        else:
            return np.clip(mean, self.control_low, self.control_high)


class iLQRPolicy(DeterministicPolicy):

    @typechecked
    def __init__(self, env_spec: EnvSpec, T: int, delta: float, iteration: int, cost_fn: CostFunc,
                 dynamics_model_train_iter: int,
                 dynamics: DynamicsEnvWrapper):
        param = Parameters(parameters=dict(T=T, delta=delta,
                                           iteration=iteration,
                                           dynamics_model_train_iter=dynamics_model_train_iter))
        super().__init__(env_spec, param)
        self.dynamics = dynamics
        self.U_hat = None
        self.X_hat = None
        self.iLqr_instance = iLQR(env_spec=env_spec,
                                  delta=self.parameters('delta'),
                                  T=self.parameters('T'),
                                  dyn_model=dynamics._dynamics,
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
        dynamics = DynamicsEnvWrapper(dynamics=self.dynamics._dynamics)
        dynamics.set_terminal_reward_func(terminal_func=self.dynamics._terminal_func,
                                          reward_func=self.dynamics._reward_func)
        return iLQRPolicy(env_spec=self.env_spec,
                          T=self.parameters('T'),
                          delta=self.parameters('delta'),
                          iteration=self.parameters('iteration'),
                          cost_fn=self.iLqr_instance.cost_fn,
                          dynamics_model_train_iter=self.parameters('dynamics_model_train_iter'),
                          dynamics=dynamics)

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
                next_obs, _, _, _ = self.dynamics.step(action=self.U_hat[i, :], state=x, allow_clip=True)
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
                x, _, _, _ = self.dynamics.step(state=x, action=u, allow_clip=True)

            X[-1] = x
            self.X_hat = X
            self.U_hat = U
        return self.U_hat[0]

    @property
    def T(self):
        return self.parameters('T')


class iLQRAlogWrapper(ModelBasedAlgo):

    def __init__(self, policy, env_spec, dynamics_env: DynamicsEnvWrapper, name: str = 'model_based_algo'):
        self.policy = policy
        super().__init__(env_spec, dynamics_env._dynamics, name)
        self.dynamics_env = dynamics_env

    def predict(self, obs, **kwargs):
        if self.is_training is True:
            return self.env_spec.action_space.sample()
        else:
            return self.policy.forward(obs=obs)

    def append_to_memory(self, *args, **kwargs):
        pass

    def init(self):
        self.policy.init()
        self.dynamics_env.init()
        super().init()

    @record_return_decorator(which_recorder='self')
    @register_counter_info_to_status_decorator(increment=1, info_key='train_counter', under_status='TRAIN')
    def train(self, *args, **kwargs) -> dict:
        super(iLQRAlogWrapper, self).train()
        res_dict = {}
        batch_data = kwargs['batch_data'] if 'batch_data' in kwargs else None
        if 'state' in kwargs:
            assert kwargs['state'] in ('state_dynamics_training', 'state_agent_training')
            state = kwargs['state']
            kwargs.pop('state')
        else:
            state = None

        if not state or state == 'state_dynamics_training':

            dynamics_train_res_dict = self._fit_dynamics_model(batch_data=batch_data,
                                                               train_iter=self.policy.parameters(
                                                                   'dynamics_model_train_iter'))
            for key, val in dynamics_train_res_dict.items():
                res_dict["{}_{}".format(self._dynamics_model.name, key)] = val
        return res_dict

    @register_counter_info_to_status_decorator(increment=1, info_key='dyanmics_train_counter', under_status='TRAIN')
    def _fit_dynamics_model(self, batch_data: TransitionData, train_iter, sess=None) -> dict:
        res_dict = self._dynamics_model.train(batch_data, **dict(sess=sess,
                                                                 train_iter=train_iter))
        return res_dict
