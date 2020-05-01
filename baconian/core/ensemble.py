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

    def __init__(self, model, n_models=1, prediction_type='random', *args, **kwargs):
        """
        :param model:
        :param n_models:
        :param prediction_type:
        """

        super().__init__(*args, **kwargs)

        self._prediction_type = prediction_type
        self._observations = list()
        self._model = list()
        self._name = kwargs.pop('name', 'dynamics_model')

        if isinstance(model, DynamicsModel):
            for a in range(n_models):
                self._model.append(model.make_copy())
        else:
            base_env_spec = model[0].env_spec
            for b in range(len(model)):
                if model[b].env_spec != base_env_spec:
                    raise TypeError('EnvSpec of list of models do not match.')
            self._model = model
        self.state = None

    def train(self, *args, **kwargs):
        res = {}
        for idx in range(len(self._model)):
            res['model_{}'.format(idx)] = self._model[idx].train(*args, **kwargs)
        return res

    def reset_state(self, *args, **kwargs):
        """
        Reset the model parameters.
        """
        self._observations = list()

        for m in self.model:
            m.reset_state(*args, **kwargs)
            self._observations.append(m.state)
        if self._prediction_type == 'mean':
            self.state = np.mean(self._observations, axis=0)
        elif self._prediction_type == 'random':
            self.state = self._observations[np.random.randint(low=0, high=len(self.model))]
        else:
            raise ValueError

    def init(self, *args, **kwargs):

        for m in self.model:
            m.init(*args, **kwargs)
        self.reset_state()

    def step(self, *args, **kwargs):

        self._observations = list()

        for m in self.model:
            self._observations.append(m.step(*args, **kwargs))

        if self._prediction_type == 'mean':
            results = np.mean(self._observations, axis=0)
        elif self._prediction_type == 'random':
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
        for model in self._model:
            model.save(*args, **kwargs)

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

    def __init__(self, cls, n_algos, prediction_type='random', *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._cls = cls
        self._n_algos = n_algos
        self._prediction_type = prediction_type
        self._predictions = list()
        self._algo = list()
        self._name = kwargs.pop('name', 'algo')

        for num in range(n_algos):
            self._algo.append(cls(*args, **dict(kwargs, name=self._name + '_' + num)))

    def init(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def algo(self):
        """
        Returns:
            The list of the models in the ensemble.
        """
        return self._algo

    def __len__(self):
        return len(self._algo)

    def __getitem__(self, idx):
        return self._algo[idx]
