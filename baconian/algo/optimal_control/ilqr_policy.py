from baconian.algo.rl.policy.policy import DeterministicPolicy
from baconian.core.parameters import Parameters
from baconian.core.core import EnvSpec
from baconian.algo.optimal_control.ilqr import iLQR as iLQR_algo
from baconian.algo.dynamics.dynamics_model import DynamicsEnvWrapper, DynamicsModel
import autograd.numpy as np
from baconian.common.special import *
from baconian.algo.dynamics.reward_func.reward_func import CostFunc
from baconian.algo.rl.rl_algo import ModelBasedAlgo
from baconian.common.logging import record_return_decorator
from baconian.core.status import register_counter_info_to_status_decorator
from baconian.core.util import init_func_arg_record_decorator
from baconian.common.sampler.sample_data import TransitionData

from baconian.core.core import Env
from baconian.algo.dynamics.dynamics_model import DynamicsEnvWrapper
"""
the gradient is computed approximated instead of analytically
"""


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
        self.iLqr_instance = iLQR_algo(env_spec=env_spec,
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
