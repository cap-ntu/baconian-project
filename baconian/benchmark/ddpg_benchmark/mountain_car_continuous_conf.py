MOUNTAIN_CAR_CONTINUOUS_BENCHMARK_CONFIG_DICT = {
    'env_id': 'MountainCarContinuous-v0',
    'MLPQValueFunction': {
        'mlp_config': [
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "1",
                "N_UNITS": 32,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "2",
                "N_UNITS": 128,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "IDENTITY",
                "B_INIT_VALUE": 0.0,
                "NAME": "OUPTUT",
                "N_UNITS": 1,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            }
        ]
    },
    'DeterministicMLPPolicy': {
        'mlp_config': [
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "1",
                "N_UNITS": 32,
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
                "ACT": "IDENTITY",
                "B_INIT_VALUE": 0.0,
                "NAME": "3",
                "N_UNITS": 1,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            }

        ]
    },
    'DDPG': {
        'config_or_config_dict': {
            "REPLAY_BUFFER_SIZE": 25000,
            "GAMMA": 0.99,
            "CRITIC_LEARNING_RATE": 0.0001,
            "ACTOR_LEARNING_RATE": 0.00001,
            "DECAY": 0.99,
            "BATCH_SIZE": 8,
            "TRAIN_ITERATION": 50,
            "critic_clip_norm": None,
            "actor_clip_norm": None,
        },
        'replay_buffer': None
    },
    'TrainTestFlow': {
        "TEST_SAMPLES_COUNT": 1,
        "TRAIN_SAMPLES_COUNT": 100,
        'config_or_config_dict': {
            "TEST_EVERY_SAMPLE_COUNT": 100,
            "TRAIN_EVERY_SAMPLE_COUNT": 100,
            "START_TRAIN_AFTER_SAMPLE_COUNT": 0,
            "START_TEST_AFTER_SAMPLE_COUNT": 0,
        }
    },
    'DEFAULT_EXPERIMENT_END_POINT': dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=300000,
                                         TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                         TOTAL_AGENT_UPDATE_COUNT=None),
}

