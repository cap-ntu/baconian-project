import gym
from autograd import grad, jacobian
import autograd.numpy as np


class ILqr:
    def __init__(self, next_state, running_cost, final_cost,
                 umax, state_dim, pred_time=50):
        self.pred_time = pred_time
        self.umax = umax
        self.v = [0.0 for _ in range(pred_time + 1)]
        self.v_x = [np.zeros(state_dim) for _ in range(pred_time + 1)]
        self.v_xx = [np.zeros((state_dim, state_dim)) for _ in range(pred_time + 1)]
        self.f = next_state
        self.lf = final_cost
        self.lf_x = grad(self.lf)
        self.lf_xx = jacobian(self.lf_x)
        self.l_x = grad(running_cost, 0)
        self.l_u = grad(running_cost, 1)
        self.l_xx = jacobian(self.l_x, 0)
        self.l_uu = jacobian(self.l_u, 1)
        self.l_ux = jacobian(self.l_u, 0)
        self.f_x = jacobian(self.f, 0)
        self.f_u = jacobian(self.f, 1)
        self.f_xx = jacobian(self.f_x, 0)
        self.f_uu = jacobian(self.f_u, 1)
        self.f_ux = jacobian(self.f_u, 0)

    def backward(self, x_seq, u_seq):
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

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        for t in range(len(u_seq)):
            control = k_seq[t] + np.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
            u_seq_hat[t] = np.clip(u_seq[t] + control, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])
        return x_seq_hat, u_seq_hat


env = gym.make('CartPoleContinuous-v0').env
obs = env.reset()
ilqr = ILqr(lambda x, u: env._state_eq(x, u),  # x(i+1) = f(x(i), u)
            lambda x, u: 0.5 * np.sum(np.square(u)),  # l(x, u)
            lambda x: 0.5 * (np.square(1.0 - np.cos(x[2])) + np.square(x[1]) + np.square(x[3])),  # lf(x)
            env.max_force,
            env.observation_space.shape[0])
u_seq = [np.zeros(1) for _ in range(ilqr.pred_time)]
x_seq = [obs.copy_from()]
for t in range(ilqr.pred_time):
    x_seq.append(env._state_eq(x_seq[-1], u_seq[t]))

cnt = 0
while True:
    env.render()
    # import pyglet
    # pyglet.image.get_buffer_manager().get_color_buffer().save('frame_%04d.png' % cnt)
    for _ in range(3):
        k_seq, kk_seq = ilqr.backward(x_seq, u_seq)
        x_seq, u_seq = ilqr.forward(x_seq, u_seq, k_seq, kk_seq)

    print(u_seq.T)
    obs, _, _, _ = env.step(u_seq[0])
    x_seq[0] = obs.copy_from()
    cnt += 1
