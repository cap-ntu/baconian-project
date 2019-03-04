"""
A simple example to show how to wrap original environment's observation space, action space and reward function for
reshaping to boost the agent's training.
Actually, this is the feature supported by gym, but since we develop a new environment class based on gym's env, so
a tutorial is given th better introduce the usage of it.
"""

from baconian.envs.env_wrapper import ObservationWrapper, ActionWrapper, RewardWrapper
from baconian.envs.gym_env import make


class SmoothMountainCarReward(RewardWrapper):

    def _reward(self, observation, action, reward, done, info):
        return 10.0


car_env = make('MountainCarContinuous-v0')
car_env = SmoothMountainCarReward(env=car_env)
car_env.reset()
new_st, reward, terminal, info = car_env.step(action=car_env.action_space.sample())
print(reward)
