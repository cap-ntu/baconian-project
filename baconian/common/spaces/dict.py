"""This is a garage-compatible wrapper for Dict spaces."""
from collections import OrderedDict

from baconian.common.spaces.base import Space
from baconian.common.spaces import Box

import numpy as np
import types


class Dict(Space):
    """
    A dictionary of simpler spaces, e.g. Discrete, Box.

    Example usage:
        self.observation_space = spaces.Dict({"position": spaces.Discrete(2),
                                              "velocity": spaces.Discrete(3)})
    """

    def __init__(self, spaces):
        """
        Convert and store the incoming spaces into an OrderedDict.

        Note: classes inheriting from garage.Dict need to convert each
        space in spaces to a garage.<class>.space.
        """
        if isinstance(spaces, dict):
            spaces = OrderedDict(sorted(list(spaces.items())))
        if isinstance(spaces, list):
            spaces = OrderedDict(spaces)
        self.spaces = spaces

    def contains(self, x):
        """
        Check if x is contained within self.spaces.

        Returns:
            Boolean

        """
        if isinstance(x, dict):
            return bool(all(item in self.spaces.items() for item in x.items()))
        else:
            return False

    def to_jsonable(self, sample_n):
        """
        Serialize as a dict-representation of vectors.

        Returns:
            JSON (dict)

        """
        return {key: space.to_jsonable([sample[key] for sample in sample_n]) \
                for key, space in self.spaces.items()}

    def from_jsonable(self, sample_n):
        """
        Convert information from a JSON format into a list.

        Returns:
            ret (list)

        """
        dict_of_list = {}
        for key, space in self.spaces.items():
            dict_of_list[key] = space.from_jsonable(sample_n[key])
        ret = []
        for i, _ in enumerate(dict_of_list[key]):
            entry = {}
            for key, value in dict_of_list.items():
                entry[key] = value[i]
            ret.append(entry)
        return ret

    @property
    def flat_dim(self):
        """
        Return a flat dimension of the dict space.

        Returns:
            flat_dim (int)

        """
        raise NotImplementedError

    def flatten(self, x):
        """
        Return a flattened observation x.

        Returns:
            x (flattened)

        """
        raise NotImplementedError

    def unflatten(self, x):
        """
        Return an unflattened observation x.

        Returns:
            x (unflattened)

        """
        raise NotImplementedError

    def flatten_n(self, xs):
        """
        Return flattened observations xs.

        Returns:
            xs (flattened)

        """
        raise NotImplementedError

    def unflatten_n(self, xs):
        """
        Return unflattened observations xs.

        Returns:
            xs (unflattened)

        """
        raise NotImplementedError

    def sample(self):
        """
        Return a sample from each space in spaces.

        Returns:
            OrderedDict

        """
        # raise NotImplementedError
        # return OrderedDict([(k, space.sample()) for k, space in self.spaces.items()])

        ordered = OrderedDict()
        for k, space in self.spaces.items():
            for a in space.low:
                if np.isinf(a):
                    a = np.nan_to_num(a)
                    space.sample = types.MethodType(self._sample_with_nan, space)
            for b in space.high:
                if np.isinf(b):
                    b = np.nan_to_num(b)
                    space.sample = types.MethodType(self._sample_with_nan, space)

            ordered.update([(k, space.sample())])
        return ordered

    def new_tensor_variable(self, name, extra_dims):
        """
        Return a new tensor variable in the TF graph.

        Returns:
            Tensor

        """
        raise NotImplementedError

    @staticmethod
    def _sample_with_nan(space: Space):
        """

        :param space:
        :return:
        """
        from gym.spaces.box import Box as GymBox

        if not isinstance(space, GymBox):
            raise TypeError('space is not of type Box')
        high = np.ones_like(space.low)
        low = -1 * np.ones_like(space.high)
        return np.clip(np.random.uniform(low=low, high=high, size=space.low.shape),
                       a_min=space.low,
                       a_max=space.high)
