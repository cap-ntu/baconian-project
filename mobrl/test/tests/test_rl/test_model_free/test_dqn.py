from mobrl.algo.rl.model_free.dqn import DQN
from mobrl.envs.gym_env import make
from mobrl.envs.env_spec import EnvSpec
from mobrl.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from mobrl.test.tests.set_up.setup import TestTensorflowSetup


class TestDQN(TestTensorflowSetup):
    def test_init(self):
        dqn, locals = self.create_dqn()
        env = locals['env']
        env_spec = locals['env_spec']
        dqn.init()
        st = env.reset()
        from mobrl.common.sampler.sample_data import TransitionData
        a = TransitionData(env_spec)
        for i in range(100):
            ac = dqn.predict(obs=st, sess=self.sess, batch_flag=False)
            st_new, re, done, _ = env.step(action=ac)
            a.append(state=st, new_state=st_new, action=ac, done=done, reward=re)
            dqn.append_to_memory(a)
            print(a.new_state_set - a.state_set)
            print(st)
        print(dqn.train(batch_data=a, train_iter=10, sess=None, update_target=True))
        print(dqn.train(batch_data=None, train_iter=10, sess=None, update_target=True))
