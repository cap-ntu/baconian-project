from mobrl.algo.rl.policy.policy import DeterministicPolicy
from mobrl.core.parameters import Parameters
from mobrl.core.core import EnvSpec
from mobrl.algo.rl.model_based.models.dynamics_model import DynamicsModel, DerivableDynamics
from typeguard import typechecked

"""
Implementation is borrowed from https://github.com/neka-nat/ilqr-gym, but rewrite with tensorflow
"""


class ILQRPolicy(DeterministicPolicy):

    @typechecked
    def __init__(self, env_spec: EnvSpec, derivable_dynamics: DynamicsModel, parameters: Parameters = None):
        super().__init__(env_spec, parameters)
        if not isinstance(derivable_dynamics, DerivableDynamics):
            raise TypeError('the dynamics must be derivable, i.e., inherited from class Derivable')
        else:
            self.dynamics = derivable_dynamics
        self.f_u = self.dynamics.grad_on_input_(self.dynamics.action_input)
        self.f_x = self.dynamics.grad_on_input_(self.dynamics.state_input)

    def forward(self, *args, **kwargs):
        pass

    def copy_from(self, obj) -> bool:
        return super().copy_from(obj)

    def make_copy(self, *args, **kwargs):
        pass

    def init(self):
        pass

    def get_status(self):
        return super().get_status()

    def _backward(self, x_seq, u_seq):
        self.v[-1] = self.lf(x_seq[-1])
        self.v_x[-1] = self.lf_x(x_seq[-1])
        self.v_xx[-1] = self.lf_xx(x_seq[-1])
        k_seq = []
        kk_seq = []
        for t in range(self.pred_time - 1, -1, -1):
            f_x_t = self.f_x(x_seq[t], u_seq[t])
            f_u_t = self.f_u(x_seq[t], u_seq[t])
            q_x = self.l_x(x_seq[t], u_seq[t]) + np.matmul(f_x_t.T, self.v_x[t + 1])
            q_u = self.l_u(x_seq[t], u_seq[t]) + np.matmul(f_u_t.T, self.v_x[t + 1])
            q_xx = self.l_xx(x_seq[t], u_seq[t]) + \
                   np.matmul(np.matmul(f_x_t.T, self.v_xx[t + 1]), f_x_t) + \
                   np.dot(self.v_x[t + 1], np.squeeze(self.f_xx(x_seq[t], u_seq[t])))
            tmp = np.matmul(f_u_t.T, self.v_xx[t + 1])
            q_uu = self.l_uu(x_seq[t], u_seq[t]) + np.matmul(tmp, f_u_t) + \
                   np.dot(self.v_x[t + 1], np.squeeze(self.f_uu(x_seq[t], u_seq[t])))
            q_ux = self.l_ux(x_seq[t], u_seq[t]) + np.matmul(tmp, f_x_t) + \
                   np.dot(self.v_x[t + 1], np.squeeze(self.f_ux(x_seq[t], u_seq[t])))
            inv_q_uu = np.linalg.inv(q_uu)
            k = -np.matmul(inv_q_uu, q_u)
            kk = -np.matmul(inv_q_uu, q_ux)
            dv = 0.5 * np.matmul(q_u, k)
            self.v[t] += dv
            self.v_x[t] = q_x - np.matmul(np.matmul(q_u, inv_q_uu), q_ux)
            self.v_xx[t] = q_xx + np.matmul(q_ux.T, kk)
            k_seq.append(k)
            kk_seq.append(kk)
        k_seq.reverse()
        kk_seq.reverse()
        return k_seq, kk_seq

    def _forward(self, x_seq, u_seq, k_seq, kk_seq):
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        for t in range(len(u_seq)):
            control = k_seq[t] + np.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
            u_seq_hat[t] = np.clip(u_seq[t] + control, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])
        return x_seq_hat, u_seq_hat
