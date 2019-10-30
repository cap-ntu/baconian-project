
ACROBOT_BENCHMARK_CONFIG_DICT = {
    'env_id': "Acrobot-v1",
    'MLPQValueFunction': {
        'mlp_config': [
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "1",
                "N_UNITS": 64,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "2",
                "N_UNITS": 64,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "RELU",
                "B_INIT_VALUE": 0.0,
                "NAME": "3",
                "N_UNITS": 256,
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
        ]
    },
    'DQN': {
        'config_or_config_dict': {
            "REPLAY_BUFFER_SIZE": 50000,
            "GAMMA": 0.99,
            "DECAY": 0,
            "BATCH_SIZE": 32,
            "TRAIN_ITERATION": 1,
            "LEARNING_RATE": 0.001,
        },
        'replay_buffer': None
    },
    'EpsilonGreedy': {
        'LinearScheduler': {
            'initial_p': 1.0,
            'final_p': 0.02,
            'schedule_timesteps': int(0.1 * 100000)
        },
        'config_or_config_dict': {
            "init_random_prob": 1.0
        }
    },
    'TrainTestFlow': {
        "TEST_SAMPLES_COUNT": 3,
        "TRAIN_SAMPLES_COUNT": 1,
        'config_or_config_dict': {
            "TEST_EVERY_SAMPLE_COUNT": 100,
            "TRAIN_EVERY_SAMPLE_COUNT": 1,
            "START_TRAIN_AFTER_SAMPLE_COUNT": 1000,
            "START_TEST_AFTER_SAMPLE_COUNT": 0,
        }
    },

    'DEFAULT_EXPERIMENT_END_POINT': dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=100000,
                                         TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                         TOTAL_AGENT_UPDATE_COUNT=None),
}
