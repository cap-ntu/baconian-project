"""
This gives a simple example on how to use Gaussian Process (GP) to approximate the Gym environment Pendulum-v0
We use gpflow package to build the Gaussian Process.
"""
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
import numpy as np
from baconian.common.sampler.sample_data import TransitionData
from baconian.algo.policy import UniformRandomPolicy
from baconian.algo.dynamics.gaussian_process_dynamiocs_model import GaussianProcessDyanmicsModel
from baconian.algo.dynamics.dynamics_model import DynamicsEnvWrapper
from baconian.algo.dynamics.terminal_func.terminal_func import RandomTerminalFunc
from baconian.algo.dynamics.reward_func.reward_func import RandomRewardFunc

env = make('Pendulum-v0')
name = 'demo_exp'
env_spec = EnvSpec(obs_space=env.observation_space,
                   action_space=env.action_space)
data = TransitionData(env_spec=env_spec)
policy = UniformRandomPolicy(env_spec=env_spec)
# Do some initial sampling here to train GP model
st = env.reset()
for i in range(100):
    ac = policy.forward(st)
    new_st, re, _, _ = env.step(ac)
    data.append(state=st, new_state=new_st, action=ac, reward=re, done=False)
    st = new_st

gp = GaussianProcessDyanmicsModel(env_spec=env_spec, batch_data=data)
gp.init()
gp.train()

dyna_env = DynamicsEnvWrapper(dynamics=gp)
# Since we only care about the prediction here, so we pass the terminal function and reward function setting with
# random one
dyna_env.set_terminal_reward_func(terminal_func=RandomTerminalFunc(),
                                  reward_func=RandomRewardFunc())

st = env.reset()
real_state_list = []
dynamics_state_list = []
test_sample_count = 100
for i in range(test_sample_count):
    ac = env_spec.action_space.sample()
    gp.reset_state(state=st)
    new_state_dynamics, _, _, _ = dyna_env.step(action=ac, allow_clip=True)
    new_state_real, _, done, _ = env.step(action=ac)
    real_state_list.append(new_state_real)
    dynamics_state_list.append(new_state_dynamics)
    st = new_state_real
    if done is True:
        env.reset()
l1_loss = np.linalg.norm(np.array(real_state_list) - np.array(dynamics_state_list), ord=1)
l2_loss = np.linalg.norm(np.array(real_state_list) - np.array(dynamics_state_list), ord=2)
print('l1 loss is {}, l2 loss is {}'.format(l1_loss, l2_loss))
