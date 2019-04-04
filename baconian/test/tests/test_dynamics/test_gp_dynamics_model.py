from baconian.test.tests.set_up.setup import TestWithAll
import numpy as np
from baconian.algo.dynamics.gaussian_process_dynamiocs_model import GaussianProcessDyanmicsModel
from baconian.common.sampler.sample_data import TransitionData
from baconian.core.core import EnvSpec
import pandas as pd


class TestDynamicsModel(TestWithAll):

    def test_dynamics_model_basic(self):
        env = self.create_env('Pendulum-v0')
        env_spec = EnvSpec(obs_space=env.observation_space, action_space=env.action_space)
        policy, _ = self.create_uniform_policy(env_spec=env_spec)
        data = TransitionData(env_spec=env_spec)
        st = env.reset()
        ac = policy.forward(st)
        for i in range(10):
            re = 0.0
            data.append(state=np.ones_like(st) * 0.5, new_state=np.ones_like(st),
                        reward=re, done=False, action=np.ones_like(ac) * 0.1)
            data.append(state=np.ones_like(st), new_state=np.ones_like(st) * 0.5,
                        reward=re, done=False, action=np.ones_like(ac) * -0.1)
        gp = GaussianProcessDyanmicsModel(env_spec=env_spec, batch_data=data)
        gp.init()
        gp.train()
        lengthscales = {}
        variances = {}
        noises = {}
        i = 0
        for model in gp.mgpr_model.models:
            lengthscales['GP' + str(i)] = model.kern.lengthscales.value
            variances['GP' + str(i)] = np.array([model.kern.variance.value])
            noises['GP' + str(i)] = np.array([model.likelihood.variance.value])
            i += 1
        print('-----Learned models------')
        pd.set_option('precision', 3)
        print('---Lengthscales---')
        print(pd.DataFrame(data=lengthscales))
        print('---Variances---')
        print(pd.DataFrame(data=variances))
        print('---Noises---')
        print(pd.DataFrame(data=noises))
        for i in range(5):
            self.assertTrue(np.isclose(gp.step(action=np.ones_like(ac) * -0.1,
                                               state=np.ones_like(st)),
                                       np.ones_like(st) * 0.5).all())
        for i in range(5):
            self.assertTrue(np.isclose(gp.step(action=np.ones_like(ac) * 0.1,
                                               state=np.ones_like(st) * 0.5),
                                       np.ones_like(st)).all())
        for i in range(5):
            print(gp.step(action=np.ones_like(ac) * -0.1,
                          state=np.ones_like(st) * 0.5))

    def test_dynamics_model_in_pendulum(self):
        env = self.create_env('Pendulum-v0')
        env_spec = EnvSpec(obs_space=env.observation_space, action_space=env.action_space)
        policy, _ = self.create_uniform_policy(env_spec=env_spec)
        data = TransitionData(env_spec=env_spec)
        st = env.reset()
        for i in range(100):
            ac = policy.forward(st)
            new_st, re, _, _ = env.step(ac)
            data.append(state=st, new_state=new_st, action=ac, reward=re, done=False)
            st = new_st

        gp = GaussianProcessDyanmicsModel(env_spec=env_spec, batch_data=data)
        gp.init()
        gp.train()
        for i in range(len(data.state_set)):
            res = gp.step(action=data.action_set[i],
                          state=data.state_set[i],
                          allow_clip=True)
            _, var = gp._state_transit(action=data.action_set[i],
                                       state=data.state_set[i],
                                       required_var=True)
            print(res)
            print(data.new_state_set[i])
            # self.assertTrue(np.isclose(res,
            #                            data.new_state_set[i], atol=1e-3).all())
            self.assertTrue(np.greater(data.new_state_set[i] + 1.96 * np.sqrt(var), res).all())
            self.assertTrue(np.less(data.new_state_set[i] - 1.96 * np.sqrt(var), res).all())

        lengthscales = {}
        variances = {}
        noises = {}
        for i, model in enumerate(gp.mgpr_model.models):
            lengthscales['GP' + str(i)] = model.kern.lengthscales.value
            variances['GP' + str(i)] = np.array([model.kern.variance.value])
            noises['GP' + str(i)] = np.array([model.likelihood.variance.value])
        print('-----Learned models------')
        pd.set_option('precision', 3)
        print('---Lengthscales---')
        print(pd.DataFrame(data=lengthscales))
        print('---Variances---')
        print(pd.DataFrame(data=variances))
        print('---Noises---')
        print(pd.DataFrame(data=noises))
