from baconian.test.tests.set_up.setup import TestWithAll
import numpy as np
from baconian.algo.dynamics.gaussian_process_dynamiocs_model import GaussianProcessDyanmicsModel
from baconian.common.sampler.sample_data import TransitionData
from baconian.core.core import EnvSpec
import pandas as pd
 

def get_some_samples(env, num, env_spec, policy):
    data = TransitionData(env_spec=env_spec)
    st = env.reset()
    for i in range(num):
        ac = policy.forward(st)
        new_st, re, _, _ = env.step(ac)
        data.append(state=st, new_state=new_st, action=ac, reward=re, done=False)
        st = new_st
    return data


class TestDynamicsModel(TestWithAll):

    # def test_more(self):
    #     for i in range(10):
    #         var = self.test_dynamics_model_in_pendulum()
    #         for v in var:
    #             del v

    def test_dynamics_model_in_pendulum(self):
        env = self.create_env('Pendulum-v0')
        env_spec = EnvSpec(obs_space=env.observation_space, action_space=env.action_space)
        policy, _ = self.create_uniform_policy(env_spec=env_spec)
        policy.allow_duplicate_name = True
        data = get_some_samples(env=env, policy=policy, num=100, env_spec=env_spec)
        gp = GaussianProcessDyanmicsModel(env_spec=env_spec, batch_data=data)
        gp.allow_duplicate_name = True
        gp.init()
        gp.train()
        print("gp first fit")
        for i in range(len(data.state_set)):
            res = gp.step(action=data.action_set[i],
                          state=data.state_set[i],
                          allow_clip=True)
            _, var = gp._state_transit(action=data.action_set[i],
                                       state=data.state_set[i],
                                       required_var=True)
            print(res)
            print(data.new_state_set[i])
            print(np.sqrt(var))
            try:
                self.assertTrue(np.isclose(res,
                                           data.new_state_set[i], atol=1e-2).all())
                self.assertTrue(np.greater_equal(data.new_state_set[i], res - 10.0 * np.sqrt(var)).all())
                self.assertTrue(np.less_equal(data.new_state_set[i], res + 10.0 * np.sqrt(var)).all())
            except Exception as e:
                print(e)
                print(np.isclose(res, data.new_state_set[i], atol=1e-2).all())
                print(np.greater_equal(data.new_state_set[i], res - 10.0 * np.sqrt(var)).all())
                print(np.less_equal(data.new_state_set[i], res + 10.0 * np.sqrt(var)).all())
                raise e

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

        # re fit the gp
        print("gp re fit")

        data = get_some_samples(env=env, policy=policy, num=100, env_spec=env_spec)
        gp.train(batch_data=data)
        for i in range(len(data.state_set)):
            res = gp.step(action=data.action_set[i],
                          state=data.state_set[i],
                          allow_clip=True)
            _, var = gp._state_transit(action=data.action_set[i],
                                       state=data.state_set[i],
                                       required_var=True)
            print(res)
            print(data.new_state_set[i])
            print(np.sqrt(var))
            try:
                self.assertTrue(np.isclose(res,
                                           data.new_state_set[i], atol=1e-1).all())
                self.assertTrue(np.greater_equal(data.new_state_set[i], res - 10.0 * np.sqrt(var)).all())
                self.assertTrue(np.less_equal(data.new_state_set[i], res + 10.0 * np.sqrt(var)).all())
            except Exception as e:
                print(e)
                print(np.isclose(res, data.new_state_set[i], atol=1e-1).all())
                print(np.greater_equal(data.new_state_set[i], res - 10.0 * np.sqrt(var)).all())
                print(np.less_equal(data.new_state_set[i], res + 10.0 * np.sqrt(var)).all())
                raise e

        # do test
        print("gp test")
        data = get_some_samples(env=env, policy=policy, num=100, env_spec=env_spec)
        for i in range(len(data.state_set)):
            res = gp.step(action=data.action_set[i],
                          state=data.state_set[i],
                          allow_clip=True)
            _, var = gp._state_transit(action=data.action_set[i],
                                       state=data.state_set[i],
                                       required_var=True)
            print(res)
            print(data.new_state_set[i])
            print(np.sqrt(var))
            print('l1 loss {}'.format(np.linalg.norm(data.new_state_set[i] - res, 1)))
        return locals()
