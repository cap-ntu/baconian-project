import numpy as np


def make_config(obs_dim, action_dim, policy_hid1_multi, value_hid3_size, value_hid1_multi, episode_per_sample,
                total_episode, episode_length=1000):
    OBS_DIM = obs_dim + 1
    HID1_MULT = value_hid1_multi
    HID3_SIZE = value_hid3_size
    HID1_SIZE = OBS_DIM * HID1_MULT
    HID2_SIZE = int(np.sqrt(HID1_SIZE * HID3_SIZE))

    POLICY_HID_MULTI = policy_hid1_multi
    ACT_DIM = action_dim
    POLICY_HID3_SIZE = ACT_DIM * 10
    POLICY_HID1_SIZE = OBS_DIM * POLICY_HID_MULTI
    POLICY_HID2_SIZE = int(np.sqrt(POLICY_HID1_SIZE * POLICY_HID3_SIZE))

    CONFIG_DICT = {
        'env_id': "HalfCheetahBulletEnv-v0",
        'MLP_V': {
            'mlp_config': [
                {
                    "ACT": "TANH",
                    "B_INIT_VALUE": 0.0,
                    "NAME": "1",
                    "N_UNITS": HID1_SIZE,
                    "TYPE": "DENSE",
                    "W_NORMAL_STDDEV": np.sqrt(1 / OBS_DIM)
                },
                {
                    "ACT": "TANH",
                    "B_INIT_VALUE": 0.0,
                    "NAME": "2",
                    "N_UNITS": HID2_SIZE,
                    "TYPE": "DENSE",
                    "W_NORMAL_STDDEV": np.sqrt(1 / HID1_SIZE)
                },
                {
                    "ACT": "TANH",
                    "B_INIT_VALUE": 0.0,
                    "NAME": "3",
                    "N_UNITS": HID3_SIZE,
                    "TYPE": "DENSE",
                    "W_NORMAL_STDDEV": np.sqrt(1 / HID2_SIZE)
                },
                {
                    "ACT": "IDENTITY",
                    "B_INIT_VALUE": 0.0,
                    "NAME": "OUPTUT",
                    "N_UNITS": 1,
                    "TYPE": "DENSE",
                    "W_NORMAL_STDDEV": np.sqrt(1 / HID3_SIZE),
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
                    "ACT": "TANH",
                    "B_INIT_VALUE": 0.0,
                    "NAME": "3",
                    "N_UNITS": POLICY_HID3_SIZE,
                    "TYPE": "DENSE",
                    "W_NORMAL_STDDEV": np.sqrt(1 / POLICY_HID2_SIZE)
                },
                {
                    "ACT": "IDENTITY",
                    "B_INIT_VALUE": 0.0,
                    "NAME": "OUPTUT",
                    "N_UNITS": ACT_DIM,
                    "TYPE": "DENSE",
                    "W_NORMAL_STDDEV": np.sqrt(1 / POLICY_HID3_SIZE)
                }
            ]
        },
        'PPO': {
            'config_or_config_dict': {
                "value_func_memory_size": int(total_episode * episode_length / 10),
                "gamma": 0.995,
                "lam": 0.98,
                "policy_train_iter": 20,
                "value_func_train_iter": 10,
                "clipping_range": None,
                "beta": 1.0,
                "eta": 50,
                "log_var_init": -1.0,
                "kl_target": 0.003,
                "policy_lr": 9e-4 / np.sqrt(POLICY_HID2_SIZE),
                "value_func_lr": 1e-2 / np.sqrt(HID2_SIZE),
                "value_func_train_batch_size": 256,
                "lr_multiplier": 1.0
            }
        },
        'TrainTestFlow': {
            "TEST_SAMPLES_COUNT": 5,
            "TRAIN_SAMPLES_COUNT": episode_per_sample,
            'config_or_config_dict': {
                "TEST_EVERY_SAMPLE_COUNT": 1,
                "TRAIN_EVERY_SAMPLE_COUNT": 1,
                "START_TRAIN_AFTER_SAMPLE_COUNT": 0,
                "START_TEST_AFTER_SAMPLE_COUNT": 0,
            }
        },
        'DEFAULT_EXPERIMENT_END_POINT': dict(
            TOTAL_AGENT_TRAIN_SAMPLE_FUNC_COUNT=int(total_episode / episode_per_sample),
            TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
            TOTAL_AGENT_UPDATE_COUNT=None),
    }
    return CONFIG_DICT
