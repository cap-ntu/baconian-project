from baconian.core.core import Basic
import abc
import numpy as np


class RewardFunc(Basic):

    def __init__(self, name='reward_func'):
        super().__init__(name=name)

    @abc.abstractmethod
    def __call__(self, state, action, new_state, **kwargs) -> float:
        raise NotImplementedError

    def init(self):
        pass


class RandomRewardFunc(RewardFunc):
    """
    Debug and test use only
    """

    def __init__(self, name='random_reward_func'):
        super().__init__(name)

    def __call__(self, state=None, action=None, new_state=None, **kwargs) -> float:
        return np.random.random()


class CostFunc(RewardFunc):
    pass


class QuadraticCostFunc(CostFunc):
    """
    A quadratic function
    """

    def __init__(self, C, c, name='reward_func'):
        """
        the cost is computed by 1/2[x_t, u_t].T *  C * [x_t, u_t] + [x_t, u_t] * c
        :param C: quadratic term
        :param c: linear term
        :param name:
        """
        super().__init__(name)
        self.C = np.array(C)
        self.c = np.array(c)
        self.state_action_flat_dim = self.C.shape[0]
        assert self.state_action_flat_dim == self.C.shape[1]
        assert len(self.C.shape) == 2
        assert self.c.shape[0] == self.state_action_flat_dim

    def __call__(self, state=None, action=None, new_state=None, **kwargs) -> float:
        u_s = np.concatenate((np.array(state).reshape(-1), np.array(action).reshape(-1))).reshape(
            self.state_action_flat_dim, 1)
        res = 0.5 * np.dot(np.dot(u_s.T, self.C), u_s) + np.dot(u_s.T, self.c).reshape(())
        return float(res)
