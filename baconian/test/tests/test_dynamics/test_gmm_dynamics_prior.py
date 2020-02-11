from baconian.test.tests.set_up.setup import TestWithAll
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.common.sampler.sample_data import TransitionData
from baconian.algo.policy import UniformRandomPolicy
from baconian.algo.dynamics.gaussian_mixture_dynamics_prior import GaussianMixtureDynamicsPrior


class TestDynamicsPrior(TestWithAll):
	def test_update(self):
		env = make('Pendulum-v0')
		name = 'demo_exp'
		env_spec = EnvSpec(obs_space=env.observation_space, action_space=env.action_space)
		data = TransitionData(env_spec=env_spec)
		policy = UniformRandomPolicy(env_spec=env_spec)

		# Do some initial sampling here to train gmm model
		st = env.reset()
		for i in range(100):
			ac = policy.forward(st)
			new_st, re, _, _ = env.step(ac)
			data.append(state=st, new_state=new_st, action=ac, reward=re, done=False)
			st = new_st

		gmm = GaussianMixtureDynamicsPrior(env_spec=env_spec, batch_data=data)
		gmm.init()
		gmm.update(batch_data=data)

	def test_prior_eval(self):
		env = make('Pendulum-v0')
		name = 'demo_exp'
		env_spec = EnvSpec(obs_space=env.observation_space, action_space=env.action_space)
		data = TransitionData(env_spec=env_spec)
		policy = UniformRandomPolicy(env_spec=env_spec)

		# Do some initial sampling here to train gmm model
		st = env.reset()
		for i in range(100):
			ac = policy.forward(st)
			new_st, re, _, _ = env.step(ac)
			data.append(state=st, new_state=new_st, action=ac, reward=re, done=False)
			st = new_st

		gmm = GaussianMixtureDynamicsPrior(env_spec=env_spec, batch_data=data)
		gmm.init()
		gmm.update(batch_data=data)
		mu0, Phi, m, n0 = gmm.eval(batch_data=data)

		state_shape = data.state_set.shape[1]
		action_shape = data.action_set.shape[1]
		self.assertEqual(state_shape + action_shape + state_shape, mu0.shape[0])
		self.assertEqual(state_shape + action_shape + state_shape, Phi.shape[0])
		self.assertEqual(state_shape + action_shape + state_shape, Phi.shape[1])
