PENDULUM_BENCHMARK_CONFIG_DICT = {
    'env_id': "Pendulum-v0",
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
                "N_UNITS": 64,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "3",
                "N_UNITS": 200,
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
                "N_UNITS": 8,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "2",
                "N_UNITS": 8,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "3",
                "N_UNITS": 8,
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
    'DDPG': {
        'config_or_config_dict': {
            "REPLAY_BUFFER_SIZE": 10000,
            "GAMMA": 0.99,
            "Q_NET_L1_NORM_SCALE": 0.01,
            "Q_NET_L2_NORM_SCALE": 0.01,
            "CRITIC_LEARNING_RATE": 0.001,
            "ACTOR_LEARNING_RATE": 0.0001,
            "DECAY": 0.5,
            "BATCH_SIZE": 128,
            "TRAIN_ITERATION": 120,
            "critic_clip_norm": None,
            "actor_clip_norm": None,
        },
        'replay_buffer': None
    },
    'EpsilonGreedy': {
        'initial_p': 1.0,
        'final_p': 0.0,
        'schedule_timesteps': 10000
    },
    'DEFAULT_EXPERIMENT_END_POINT': dict(TOTAL_ENV_STEP_TRAIN_SAMPLE_COUNT=10000),
    'DynaFlow': {
        "TEST_ALGO_EVERY_REAL_SAMPLE_COUNT": 10,
        "TEST_DYNAMICS_EVERY_REAL_SAMPLE_COUNT": 100,
        "TRAIN_DYNAMICS_EVERY_REAL_SAMPLE_COUNT": 10,
        "START_TRAIN_ALGO_AFTER_SAMPLE_COUNT": 1,
        "START_TRAIN_DYNAMICS_AFTER_SAMPLE_COUNT": 1,
        "START_TEST_ALGO_AFTER_SAMPLE_COUNT": 1,
        "START_TEST_DYNAMICS_AFTER_SAMPLE_COUNT": 1,
        "WARM_UP_DYNAMICS_SAMPLES": 8000,
        "TRAIN_ALGO_EVERY_REAL_SAMPLE_COUNT_FROM_REAL_ENV": 10,
        "TRAIN_ALGO_EVERY_REAL_SAMPLE_COUNT_FROM_DYNAMICS_ENV": 100,
    },
    'DynamicsModel': dict(learning_rate=0.01,
                          mlp_config=[
                              {
                                  "ACT": "RELU",
                                  "B_INIT_VALUE": 0.0,
                                  "NAME": "1",
                                  "N_UNITS": 32,
                                  "TYPE": "DENSE",
                                  "W_NORMAL_STDDEV": 0.03
                              },
                              {
                                  "ACT": "RELU",
                                  "B_INIT_VALUE": 0.0,
                                  "NAME": "2",
                                  "N_UNITS": 64,
                                  "TYPE": "DENSE",
                                  "W_NORMAL_STDDEV": 0.03
                              },
                              {
                                  "ACT": "IDENTITY",
                                  "B_INIT_VALUE": 0.0,
                                  "NAME": "OUPTUT",
                                  "N_UNITS": 3,
                                  "TYPE": "DENSE",
                                  "W_NORMAL_STDDEV": 0.03
                              }
                          ])

}
