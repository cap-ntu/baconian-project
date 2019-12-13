from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
import numpy as np

env = make("HalfCheetah-v2")
env_spec = EnvSpec(obs_space=env.observation_space, action_space=env.action_space)

OBS_DIM = env_spec.flat_obs_dim
HID1_SIZE = 400
HID2_SIZE = 300

POLICY_HID_MULTI = 10
ACT_DIM = env_spec.flat_action_dim
POLICY_HID1_SIZE = 400
POLICY_HID2_SIZE = 300

CHEETAH_BENCHMARK_CONFIG_DICT = {
    'env_id': "HalfCheetah-v2",
    'MLP_V': {
        'mlp_config': [
            {
                "ACT": "RELU",
                "B_INIT_VALUE": 0.0,
                "NAME": "1",
                "N_UNITS": HID1_SIZE,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": np.sqrt(1 / OBS_DIM)
            },
            {
                "ACT": "RELU",
                "B_INIT_VALUE": 0.0,
                "NAME": "2",
                "N_UNITS": HID2_SIZE,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": np.sqrt(1 / HID1_SIZE)
            },
            {
                "ACT": "IDENTITY",
                "B_INIT_VALUE": 0.0,
                "NAME": "OUPTUT",
                "N_UNITS": 1,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": np.sqrt(1 / HID2_SIZE),
            }
        ]
    },
    'POLICY': {
        'mlp_config': [
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "1",
                "N_UNITS": POLICY_HID1_SIZE,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": np.sqrt(1 / OBS_DIM)
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "2",
                "N_UNITS": POLICY_HID2_SIZE,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": np.sqrt(1 / POLICY_HID1_SIZE)
            },
            {
                "ACT": "IDENTITY",
                "B_INIT_VALUE": 0.0,
                "NAME": "OUPTUT",
                "N_UNITS": ACT_DIM,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": np.sqrt(1 / POLICY_HID2_SIZE)
            }
        ]
    },
    'DDPG': {
        'config_or_config_dict': {
            "REPLAY_BUFFER_SIZE": 1000000,
            "GAMMA": 0.99,
            "CRITIC_LEARNING_RATE": 0.001,
            "ACTOR_LEARNING_RATE": 0.0001,
            "DECAY": 0.999,
            "BATCH_SIZE": 128,
            "TRAIN_ITERATION": 120,
            "critic_clip_norm": None,
            "actor_clip_norm": None,
        },
        'replay_buffer': None
    },
    'TrainTestFlow': {
        "TEST_SAMPLES_COUNT": 10,
        "TRAIN_SAMPLES_COUNT": 1,
        'config_or_config_dict': {
            "TEST_EVERY_SAMPLE_COUNT": 1000,
            "TRAIN_EVERY_SAMPLE_COUNT": 1,
            "START_TRAIN_AFTER_SAMPLE_COUNT": 10000,
            "START_TEST_AFTER_SAMPLE_COUNT": 20,
        }
    },

    'DEFAULT_EXPERIMENT_END_POINT': dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=1000000,
                                         TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                         TOTAL_AGENT_UPDATE_COUNT=None),
    'AGENT_NOISE': {

    }
}