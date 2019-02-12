import unittest
from src.envs.gym_env import make
from src.envs.env_spec import EnvSpec
import tensorflow as tf
from src.tf.util import create_new_tf_session
from src.common.sampler.sample_data import TransitionData, TrajectoryData
from src.rl.value_func.mlp_v_value import MLPVValueFunc
from src.rl.policy.normal_distribution_mlp import NormalDistributionMLPPolicy
from src.rl.algo.model_free.ppo import PPO


class TestPPO(unittest.TestCase):
    def test_init(self):
        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
            # sess.close()
        tf.reset_default_graph()
        env = make('Swimmer-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        sess = create_new_tf_session(cuda_device=1)

        mlp_v = MLPVValueFunc(env_spec=env_spec,
                              name_scope='mlp_v',
                              mlp_config=[
                                  {
                                      "ACT": "RELU",
                                      "B_INIT_VALUE": 0.0,
                                      "NAME": "1",
                                      "N_UNITS": 16,
                                      "TYPE": "DENSE",
                                      "W_NORMAL_STDDEV": 0.03
                                  },
                                  {
                                      "ACT": "LINEAR",
                                      "B_INIT_VALUE": 0.0,
                                      "NAME": "OUPTUT",
                                      "N_UNITS": 1,
                                      "TYPE": "DENSE",
                                      "W_NORMAL_STDDEV": 0.03
                                  }
                              ])
        policy = NormalDistributionMLPPolicy(env_spec=env_spec,
                                             name_scope='mlp_policy',
                                             mlp_config=[
                                                 {
                                                     "ACT": "RELU",
                                                     "B_INIT_VALUE": 0.0,
                                                     "NAME": "1",
                                                     "N_UNITS": 16,
                                                     "TYPE": "DENSE",
                                                     "W_NORMAL_STDDEV": 0.03
                                                 },
                                                 {
                                                     "ACT": "LINEAR",
                                                     "B_INIT_VALUE": 0.0,
                                                     "NAME": "OUPTUT",
                                                     "N_UNITS": env_spec.flat_action_dim,
                                                     "TYPE": "DENSE",
                                                     "W_NORMAL_STDDEV": 0.03
                                                 }
                                             ],
                                             reuse=False)
        ppo = PPO(
            env_spec=env_spec,
            config_or_config_dict={
                "gamma": 0.995,
                "lam": 0.98,
                "policy_train_iter": 10,
                "value_func_train_iter": 10,
                "clipping_range": None,
                "beta": 1.0,
                "eta": 50,
                "log_var_init": -1.0,
                "kl_target": 0.003,
                "policy_lr": 0.01,
                "value_func_lr": 0.01,
                "value_func_train_batch_size": 10
            },
            value_func=mlp_v,
            stochastic_policy=policy,
            adaptive_learning_rate=True,
            name='ppo',
        )
        ppo.init()
        print(tf.report_uninitialized_variables())
        data = TransitionData(env_spec)
        st = env.reset()
        for i in range(100):
            ac = ppo.predict(st)
            assert ac.shape[0] == 1
            self.assertTrue(env_spec.action_space.contains(ac[0]))
            new_st, re, done, _ = env.step(ac)
            if i == 99:
                done = True
            data.append(state=st, new_state=new_st, action=ac, reward=re, done=done)
        ppo.append_to_memory(data)
        print(ppo.train())

        traj_data = TrajectoryData(env_spec=env_spec)
        traj_data.append(data)
        print(
            ppo.train(trajectory_data=traj_data,
                      train_iter=10,
                      sess=sess))
        # ppo.append_to_memory(data)
        # for i in range(1000):
        #     print(ppo.train())


if __name__ == '__main__':
    TestPPO()
