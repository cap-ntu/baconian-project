from baconian.core.core import Basic, Env
from baconian.algo.dynamics.dynamics_model import DynamicsModel, DynamicsEnvWrapper
from baconian.algo.algo import Algo
import numpy as np
import abc


class Ensemble(Basic):

    def init(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError


class ModelEnsemble(Ensemble, DynamicsModel):

    def __init__(self, n_models=1, prediction='random', *args, **kwargs):
        """
        :param env_spec:
        :param parameters:
        :param init_state:
        :param name:
        :param n_models:
        :param prediction:
        """

        super().__init__(*args, **kwargs)
        self._prediction = prediction
        self._observations = list()
        self._model = list()
        self._name = kwargs.pop('name', 'dynamics_model')

        for num in range(n_models):
            self._model.append(DynamicsModel(*args, **dict(kwargs, name=self._name + '_' + num)))

    def fit(self, *z, **fit_params):
        """
        Fit the ``idx``-th model of the ensemble if ``idx`` is provided, a
        random model otherwise.
        Args:
            *z (list): a list containing the inputs to use to predict with each
                regressor of the ensemble;
            **fit_params (dict): other params.
        """
        idx = fit_params.pop('idx', None)
        if idx is None:
            self[np.random.choice(len(self))].fit(*z, **fit_params)
        else:
            self[idx].fit(*z, **fit_params)

    def reset_state(self, *args, **kwargs):
        """
        Reset the model parameters.
        """
        for m in self.model:
            m.reset_state(*args, **kwargs)

    def init(self, *args, **kwargs):

        for m in self.model:
            m.init(*args, **kwargs)

    def step(self, *args, **kwargs):

        self._observations = list()

        for m in self.model:
            self._observations.append(m.step(*args, **kwargs))

        if self._prediction == 'mean':
            results = np.mean(self._observations, axis=0)
        elif self._prediction == 'sum':
            results = np.sum(self._observations, axis=0)
        elif self._prediction == 'random':
            results = self._observations[np.random.randint(low=0, high=len(self.model))]
        else:
            raise ValueError

        return results

    def get_obs(self):
        return self._observations

    @abc.abstractmethod
    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def copy_from(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        return True

    def make_copy(self):
        raise NotImplementedError

    def return_as_env(self) -> Env:
        return DynamicsEnvWrapper(dynamics=self,
                                  name=self._name + '_env')

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def model(self):
        """
        Returns:
            The list of the models in the ensemble.
        """
        return self._model

    def __len__(self):
        return len(self._model)

    def __getitem__(self, idx):
        return self._model[idx]


class AlgoEnsemble(Ensemble, Algo):

    def init(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError

