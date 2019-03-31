MOUNTAIN_CAR_CONTINUOUS_BENCHMARK_CONFIG_DICT = {
    'env_id': 'MountainCarContinuous-v0',
    'MLPQValueFunction': {
        'mlp_config': [
            {
                "ACT": "ELU",
                "B_INIT_VALUE": 0.0,
                "NAME": "1",
                "N_UNITS": 32,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "ELU",
                "B_INIT_VALUE": 0.0,
                "NAME": "2",
                "N_UNITS": 128,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "IDENTITY",
                "B_INIT_VALUE": 0.0,
                "NAME": "OUTPUT",
                "N_UNITS": 1,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            }
        ]
    },
    'DeterministicMLPPolicy': {
        'mlp_config': [
            {
                "ACT": "ELU",
                "B_INIT_VALUE": 0.0,
                "NAME": "1",
                "N_UNITS": 32,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "ELU",
                "B_INIT_VALUE": 0.0,
                "NAME": "2",
                "N_UNITS": 64,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": None,
                "NAME": "OUTPUT",
                "N_UNITS": 1,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            }
        ]
    },
    'DDPG': {
        'config_or_config_dict': {
            "REPLAY_BUFFER_SIZE": 10000,
            "GAMMA": 0.99,
            "Q_NET_L1_NORM_SCALE": 0.,
            "Q_NET_L2_NORM_SCALE": 0.,
            "CRITIC_LEARNING_RATE": 0.001,
            "ACTOR_LEARNING_RATE": 0.0001,
            "DECAY": 0.5,
            "BATCH_SIZE": 256,
            "CRITIC_TRAIN_ITERATION": 1,
            "ACTOR_TRAIN_ITERATION": 1,
            "critic_clip_norm": 0.0,
            "actor_clip_norm": 0.0,
        },
        'adaptive_learning_rate': False,
        'replay_buffer': None
    },
    'Agent': {
        'config_or_config_dict': {
            "TEST_SAMPLES_COUNT": 2000,
            "TRAIN_SAMPLES_COUNT": 1,
            "TOTAL_SAMPLES_COUNT": 30000
        },
    },
    'EpsilonGreedy': {
        'initial_p': 1.0,
        'final_p': 0.0,
        'schedule_timesteps': 15000.0
    }
}
