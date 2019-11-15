from baconian.core.core import EnvSpec
from baconian.algo.dynamics.dynamics_model import DynamicsPriorModel
import numpy as np
from math import inf
from baconian.common.sampler.sample_data import TransitionData
from baconian.algo.dynamics.third_party.gmm import GMM
import tensorflow as tf
from baconian.tf.util import *
from baconian.tf.tf_parameters import ParametersWithTensorflowVariable
from baconian.common.data_pre_processing import DataScaler

class GaussianMixtureDynamicsPrior(DynamicsPriorModel):
    """
    A dynamics prior encoded as a GMM over [x_t, u_t, x_t+1] points.
    See:
        S. Levine*, C. Finn*, T. Darrell, P. Abbeel, "End-to-end
        training of Deep Visuomotor Policies", arXiv:1504.00702,
        Appendix A.3.
    """
    def __init__(self, env_spec: EnvSpec, batch_data: TransitionData = None, epsilon=inf, init_sequential=False, eigreg=False, warmstart=True, init_state=None, name_scope='gp_dynamics_model', name='gp_dynamics_model'):
        parameters = ParametersWithTensorflowVariable(tf_var_list=[],
                                                      rest_parameters=dict(),
                                                      name='{}_param'.format(name),
                                                      require_snapshot=False)
        
        super().__init__(env_spec=env_spec, parameters=parameters, init_state=init_state, name=name)
        self.name_scope = name_scope
        self.X = None
        self.U = None
        self.batch_data = batch_data
        self.gmm_model = GMM(epsilon=epsilon, init_sequential=False, eigreg=False, warmstart=True)
        self._min_samp = 40
        self._max_samples = inf
        self._max_clusters = 20
        self._strength = 1

    def init(self):
        super().init()
    
    def update(self, restart=1, batch_data: TransitionData = None, *kwargs):
        """
        Update prior with additional data.
        Args:
            X: A N x T x dX matrix of sequential state data.
            U: A N x T x dU matrix of sequential control data.
        """
        # Format Data
        xux, K = self.prepare_data(batch_data)

        # Update GMM.
        self.gmm_model.update(xux, K)

    def _state_transit(self, state, action, required_var=False, **kwargs):
        # TODO: GETTING x(t+1) from x(t), u(t)
        # Use np.random.multivariate_normal(means, covariances, samples)
        return state

    def eval(self, batch_data: TransitionData = None):
        """
        Evaluate prior (prob of [x(t), u(t), x(t+1)] given gmm)
        Args:
            batch_data: A N x Dx+Du+Dx matrix.
        """
        # Format Data
        xux, _ = self.prepare_data(batch_data)

        # Perform query and fix mean.
        mu0, Phi, m, n0 = self.gmm_model.inference(xux)

        # Factor in multiplier.
        n0 = n0 * self._strength
        m = m * self._strength

        # Multiply Phi by m (since it was normalized before).
        Phi *= m
        return mu0, Phi, m, n0

    def prepare_data(self, batch_data: TransitionData = None, *kwargs):
        """
        Formats Data into the correct shape and dimensions for feeding into gmm
        Args:
            X: A N x T x dX matrix of sequential state data.
            U: A N x T x dU matrix of sequential control data.
        Returns:
            xux: A T*N x Do matrix of [X(t), U(t), X(t+1)] data
        """
        if self.batch_data is None:
            self.batch_data = batch_data

        X = batch_data.state_set
        if X.ndim == 2: X = np.expand_dims(X, axis=0)
        U = batch_data.action_set
        if U.ndim == 2: U = np.expand_dims(U, axis=0)

        # Constants.
        T = X.shape[1] - 1

        # Append data to dataset.
        if self.X is None:
            self.X = X
        else:
            self.X = np.concatenate([self.X, X], axis=0)

        if self.U is None:
            self.U = U
        else:
            self.U = np.concatenate([self.U, U], axis=0)

        # Remove excess samples from dataset.
        start = max(0, self.X.shape[0] - self._max_samples + 1)
        self.X = self.X[start:, :]
        self.U = self.U[start:, :]

        # Compute cluster dimensionality.
        Do = X.shape[2] + U.shape[2] + X.shape[2]  #TODO: Use Xtgt.

        # Create dataset.
        N = self.X.shape[0]
        xux = np.reshape(
            np.c_[self.X[:, :T, :], self.U[:, :T, :], self.X[:, 1:(T+1), :]],
            [T * N, Do]
        )

        # Choose number of clusters.
        K = int(max(2, min(self._max_clusters, np.floor(float(N * T) / self._min_samp))))

        return xux, K