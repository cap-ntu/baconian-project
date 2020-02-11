from baconian.core.core import EnvSpec
from baconian.algo.dynamics.dynamics_model import DynamicsPriorModel
import numpy as np
from math import inf
from baconian.common.sampler.sample_data import TransitionData
from baconian.algo.dynamics.third_party.gmm import GMM
from baconian.core.parameters import Parameters


class GaussianMixtureDynamicsPrior(DynamicsPriorModel):
    """
    A dynamics prior encoded as a GMM over [x_t, u_t, x_t+1] points.
    See:
        S. Levine*, C. Finn*, T. Darrell, P. Abbeel, "End-to-end
        training of Deep Visuomotor Policies", arXiv:1504.00702,
        Appendix A.3.
    """

    def __init__(self, env_spec: EnvSpec, batch_data: TransitionData = None, epsilon=inf, init_sequential=False,
                 eigreg=False, warmstart=True, name_scope='gp_dynamics_model', min_samples_per_cluster=40,
                 max_clusters=20, strength=1,
                 name='gp_dynamics_model'):
        parameters = Parameters(
            dict(min_samp=min_samples_per_cluster, max_samples=inf, max_clusters=max_clusters, strength=strength,
                 init_sequential=init_sequential, eigreg=eigreg, warmstart=warmstart))
        super().__init__(env_spec=env_spec, parameters=parameters, name=name)
        self.name_scope = name_scope
        self.batch_data = batch_data
        self.gmm_model = GMM(epsilon=epsilon, init_sequential=init_sequential, eigreg=eigreg, warmstart=warmstart)
        self.X, self.U = None, None

    def init(self):
        pass

    def update(self, batch_data: TransitionData = None):
        """
        Update prior with additional data.
        :param batch_data: data used to update GMM prior
        :return: None
        """

        # Format Data
        xux, K = self._prepare_data(batch_data)

        # Update GMM.
        self.gmm_model.update(xux, K)

    def eval(self, batch_data: TransitionData = None):
        """
        Evaluate prior (prob of [x(t), u(t), x(t+1)] given gmm)

        :param batch_data: data used to evaluate the prior with.
        :return: parameters mu0, Phi, m, n0 as defined in the paper.
        """
        # Format Data
        xux, _ = self._prepare_data(batch_data)

        # Perform query and fix mean.
        mu0, Phi, m, n0 = self.gmm_model.inference(xux)

        # Factor in multiplier.
        n0 = n0 * self.parameters('strength')
        m = m * self.parameters('strength')

        # Multiply Phi by m (since it was normalized before).
        Phi *= m
        return mu0, Phi, m, n0

    def _prepare_data(self, batch_data: TransitionData = None, *kwargs):
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
        start = max(0, self.X.shape[0] - self.parameters('max_samples') + 1)
        self.X = self.X[start:, :]
        self.U = self.U[start:, :]

        # Compute cluster dimensionality.
        Do = X.shape[2] + U.shape[2] + X.shape[2]  # TODO: Use Xtgt.

        # Create dataset.
        N = self.X.shape[0]
        xux = np.reshape(
            np.c_[self.X[:, :T, :], self.U[:, :T, :], self.X[:, 1:(T + 1), :]],
            [T * N, Do]
        )

        # Choose number of clusters.
        K = int(max(2, min(self.parameters('max_clusters'), np.floor(float(N * T) / self.parameters('min_samp')))))

        return xux, K
