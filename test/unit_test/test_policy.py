import unittest
from src.model.policy.policy import Policy
from gym.core import Space
from gym.spaces.discrete import Discrete


class TestPolicy(unittest.TestCase):
    def test_init(self):
        a = Policy(action_space=Space(), obs_space=Space())
        a = Policy(action_space=Discrete(10), obs_space=Discrete(10))
        try:
            a = Policy(action_space=None, obs_space=Space())
        except TypeError as e:
            pass


if __name__ == '__main__':
    unittest.main()
