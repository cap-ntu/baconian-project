from gym import make
from src.envs.env_spec import EnvSpec
import tensorflow as tf
from src.tf.util import create_new_tf_session
from src.rl.algo.model_based.models.mlp_dynamics_model import ContinuousMLPDynamicsModel
from src.common.sampler.sample_data import TransitionData
from src.rl.algo.model_based.mpc import ModelPredictiveControl
import unittest
from src.rl.algo.model_based.misc.terminal_func.terminal_func import RandomTerminalFunc
from src.rl.algo.model_based.misc.reward_func.reward_func import RandomRewardFunc


class TestMPC(unittest.TestCase):

    def test_init(self):
        if tf.get_default_session():
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
            # sess.close()
        tf.reset_default_graph()
        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)
        sess = create_new_tf_session(cuda_device=0)

        mlp_dyna = ContinuousMLPDynamicsModel(
            env_spec=env_spec,
            name_scope='mlp_dyna',
            input_norm=False,
            output_norm=False,
            output_low=env_spec.obs_space.low,
            output_high=env_spec.obs_space.high,
            l1_norm_scale=1.0,
            l2_norm_scale=1.0,
            learning_rate=0.01,
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
                    "N_UNITS": env_spec.flat_obs_dim,
                    "TYPE": "DENSE",
                    "W_NORMAL_STDDEV": 0.03
                }
            ])
        algo = ModelPredictiveControl(
            dynamics_model=mlp_dyna,
            env_spec=env_spec,
            config_or_config_dict=dict(
                SAMPLED_HORIZON=2,
                SAMPLED_PATH_NUM=5,
                dynamics_model_train_iter=10
            ),
            reward_func=RandomRewardFunc(),
            terminal_func=RandomTerminalFunc()
        )
        algo.init()
        for _ in range(100):
            assert env_spec.action_space.contains(algo.predict(env_spec.obs_space.sample()))

        st = env.reset()
        data = TransitionData(env_spec)

        for _ in range(10):
            ac = algo.predict(st)
            new_st, re, done, _ = env.step(action=ac)
            data.append(state=st,
                        new_state=new_st,
                        reward=re,
                        action=ac,
                        done=done)
        print(algo.train(batch_data=data))


if __name__ == '__main__':
    unittest.main()
