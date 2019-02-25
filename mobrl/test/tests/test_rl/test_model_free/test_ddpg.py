from mobrl.envs.gym_env import make
from mobrl.envs.env_spec import EnvSpec
from mobrl.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from mobrl.algo.rl.model_free.ddpg import DDPG
from mobrl.algo.rl.policy.deterministic_mlp import DeterministicMLPPolicy
from mobrl.common.sampler.sample_data import TransitionData
from mobrl.test.tests.test_setup import TestTensorflowSetup


class TestDDPG(TestTensorflowSetup):
    def test_init(self):
        env = make('Swimmer-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        mlp_q = MLPQValueFunction(env_spec=env_spec,
                                  name_scope='mlp_q',
                                  name='mlp_q',
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
        policy = DeterministicMLPPolicy(env_spec=env_spec,
                                        name_scope='mlp_policy',
                                        name='mlp_policy',
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
        ddpg = DDPG(
            env_spec=env_spec,
            config_or_config_dict={
                "REPLAY_BUFFER_SIZE": 10000,
                "GAMMA": 0.999,
                "Q_NET_L1_NORM_SCALE": 0.01,
                "Q_NET_L2_NORM_SCALE": 0.01,
                "CRITIC_LEARNING_RATE": 0.001,
                "ACTOR_LEARNING_RATE": 0.001,
                "DECAY": 0.5,
                "ACTOR_BATCH_SIZE": 5,
                "CRITIC_BATCH_SIZE": 5,
                "CRITIC_TRAIN_ITERATION": 1,
                "ACTOR_TRAIN_ITERATION": 1,
                "critic_clip_norm": 0.001
            },
            value_func=mlp_q,
            policy=policy,
            adaptive_learning_rate=True,
            name='ddpg',
            replay_buffer=None
        )
        ddpg.init()
        data = TransitionData(env_spec)
        st = env.reset()
        for i in range(100):
            ac = ddpg.predict(st)
            new_st, re, done, _ = env.step(ac)
            data.append(state=st, new_state=new_st, action=ac, reward=re, done=done)
        ddpg.append_to_memory(data)
        for i in range(10):
            print(ddpg.train())
