from baconian.algo.algo import Algo
from baconian.algo.dynamics.dynamics_model import DynamicsModel
from baconian.core.core import EnvSpec
from baconian.common.logging import record_return_decorator
import numpy as np


class ModelFreeAlgo(Algo):
    def __init__(self, env_spec: EnvSpec, name: str = 'model_free_algo', warm_up_trajectories_number=0):
        super(ModelFreeAlgo, self).__init__(env_spec, name, warm_up_trajectories_number)


class OnPolicyAlgo(Algo):
    pass


class OffPolicyAlgo(Algo):
    pass


class ValueBasedAlgo(Algo):
    pass


class PolicyBasedAlgo(Algo):
    pass


class ModelBasedAlgo(Algo):
    def __init__(self, env_spec, dynamics_model: DynamicsModel, name: str = 'model_based_algo'):
        super(ModelBasedAlgo, self).__init__(env_spec, name)
        self._dynamics_model = dynamics_model
        self.dynamics_env = self._dynamics_model.return_as_env()

    def train_dynamics(self, *args, **kwargs):
        pass

    @record_return_decorator(which_recorder='self')
    def test_dynamics(self, env, sample_count, *args, **kwargs):
        self.set_status('TEST')
        env.set_status('TEST')
        st = env.reset()
        real_state_list = []
        dyanmics_state_list = []
        for i in range(sample_count):
            ac = self.env_spec.action_space.sample()
            self._dynamics_model.reset_state(state=st)
            new_state_dynamics, _, _, _ = self.dynamics_env.step(action=ac, )
            new_state_real, _, done, _ = env.step(action=ac)
            real_state_list.append(new_state_real)
            dyanmics_state_list.append(new_state_dynamics)
            st = new_state_real
            if done is True:
                env.reset()
        l1_loss = np.linalg.norm(np.array(real_state_list) - np.array(dyanmics_state_list), ord=1)
        l2_loss = np.linalg.norm(np.array(real_state_list) - np.array(dyanmics_state_list), ord=2)
        return dict(dynamics_test_l1_error=l1_loss, dynamics_test_l2_error=l2_loss)

    def set_terminal_reward_function_for_dynamics_env(self, terminal_func, reward_func):
        self.dynamics_env.set_terminal_reward_func(terminal_func, reward_func)
