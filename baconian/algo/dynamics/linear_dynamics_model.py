from baconian.common.data_pre_processing import DataScaler
from baconian.core.core import EnvSpec
from baconian.algo.dynamics.dynamics_model import GlobalDynamicsModel, TrainableDyanmicsModel

from baconian.core.parameters import Parameters
import numpy as np
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from baconian.common.sampler.sample_data import TransitionData


class LinearDynamicsModel(GlobalDynamicsModel):
    """
    A linear dynamics model given the transition matrix F and the bias f (can't be trained, use LinearRegressionModel instead if your want to fit one)
    """

    def __init__(self, env_spec: EnvSpec, state_transition_matrix: np.array, bias: np.array, init_state=None,
                 name='dynamics_model', state_input_scaler: DataScaler = None, action_input_scaler: DataScaler = None,
                 state_output_scaler: DataScaler = None):
        parameters = Parameters(parameters=dict(F=state_transition_matrix, f=bias))
        super().__init__(env_spec, parameters, init_state, name, state_input_scaler, action_input_scaler,
                         state_output_scaler)

        assert self.parameters('F').shape == \
               (env_spec.obs_space.flat_dim, env_spec.obs_space.flat_dim + env_spec.action_space.flat_dim)
        assert self.parameters('f').shape[0] == env_spec.obs_space.flat_dim

    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        new_state = np.dot(self.parameters('F'), np.concatenate((state, action))) + self.parameters('f')
        return self.env_spec.obs_space.clip(new_state)

    def make_copy(self):
        return LinearDynamicsModel(env_spec=self.env_spec,
                                   state_transition_matrix=deepcopy(self.parameters('F')),
                                   bias=deepcopy(self.parameters('f')))

    @property
    def F(self):
        return self.parameters('F')

    @property
    def f(self):
        return self.parameters('f')


class LinearRegressionDynamicsModel(GlobalDynamicsModel, TrainableDyanmicsModel):

    def __init__(self, env_spec: EnvSpec, init_state=None, name='dynamics_model',
                 state_input_scaler: DataScaler = None, action_input_scaler: DataScaler = None,
                 state_output_scaler: DataScaler = None):
        super().__init__(env_spec=env_spec, init_state=init_state, name=name,
                         state_input_scaler=state_input_scaler,
                         action_input_scaler=action_input_scaler,
                         state_output_scaler=state_output_scaler)
        self._linear_model = LinearRegression(fit_intercept=True,
                                              normalize=False)

    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        state = self.state_input_scaler.process(np.array(state).reshape(self.env_spec.obs_shape))
        action = self.action_input_scaler.process(action.reshape(self.env_spec.flat_action_dim))
        new_state = self._linear_model.predict(np.concatenate([state, action], axis=-1).reshape(1, -1))
        new_state = np.clip(self.state_output_scaler.inverse_process(new_state),
                            self.env_spec.obs_space.low,
                            self.env_spec.obs_space.high).squeeze()
        return new_state

    def train(self, batch_data: TransitionData = None, *kwargs):
        self.state_input_scaler.update_scaler(batch_data.state_set)
        self.action_input_scaler.update_scaler(batch_data.action_set)
        self.state_output_scaler.update_scaler(batch_data.new_state_set)

        state = self.state_input_scaler.process(batch_data.state_set)
        action = self.action_input_scaler.process(batch_data.action_set)
        new_state = self.state_input_scaler.process(batch_data.new_state_set)

        self._linear_model.fit(X=np.concatenate([state, action], axis=-1),
                               y=new_state)
