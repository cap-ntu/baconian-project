from baconian.core.core import EnvSpec
from baconian.algo.dynamics.dynamics_model import DynamicsModel, TrainableDyanmicsModel
import gpflow
from baconian.core.parameters import Parameters
import numpy as np
from copy import deepcopy
from baconian.common.sampler.sample_data import TransitionData
from baconian.algo.dynamics.third_party.mgpr import MGPR
import tensorflow as tf
from baconian.tf.util import *
from baconian.tf.tf_parameters import ParametersWithTensorflowVariable


class GaussianProcessDyanmicsModel(TrainableDyanmicsModel):
    kernel_type_dict = {
        'RBF': (gpflow.kernels.RBF, dict(ARD=True))
    }

    """
    Dynamics approximated by multivariate gaussian process model based GPflow package.
    Mostly refer the implementation of  PILCO repo in https://github.com/nrontsis/PILCO
    """

    def __init__(self, env_spec: EnvSpec, init_state=None,
                 name_scope='gp_dynamics_model', name='gp_dynamics_model',
                 gp_kernel_type='RBF'):
        if gp_kernel_type not in self.kernel_type_dict.keys():
            raise TypeError(
                'Not supported {} kernel, choose from'.format(gp_kernel_type, list(self.kernel_type_dict.keys())))
        parameters = ParametersWithTensorflowVariable(tf_var_list=[],
                                                      rest_parameters=dict(),
                                                      name='{}_param'.format(name),
                                                      require_snapshot=False)
        super().__init__(env_spec, parameters, init_state, name)
        self.name_scope = name_scope
        self.mgpr_model = MGPR(name='mgpr', action_dim=env_spec.flat_action_dim, state_dim=env_spec.flat_obs_dim)

    def init(self, batch_data: TransitionData, **kwargs):
        state_action_data = np.hstack((batch_data.state_set, batch_data.action_set))
        delta_state_data = batch_data.new_state_set - batch_data.state_set
        with tf.variable_scope(self.name_scope):
            self.mgpr_model.init(X=state_action_data, Y=delta_state_data)
        var_list = get_tf_collection_var_list(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope=self.name_scope)
        self.parameters.set_tf_var_list(tf_var_list=sorted(list(set(var_list)), key=lambda x: x.name))
        super().init()

    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        deltas, vars = self.mgpr_model.predict(x=np.concatenate([state, action], axis=-1))
        return deltas + state

    def copy_from(self, obj) -> bool:
        raise NotImplementedError
        # return super().copy_from(obj)

    def make_copy(self):
        raise NotImplementedError

    def train(self, restart=1, batch_data: TransitionData = None, *kwargs):
        if batch_data:
            state_action_data = np.hstack((batch_data.state_set, batch_data.action_set))
            delta_state_data = batch_data.new_state_set - batch_data.state_set
            self.mgpr_model.set_XY(X=state_action_data, Y=delta_state_data)
        self.mgpr_model.optimize(restarts=restart)
