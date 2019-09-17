INVERTED_PENDULUM_BENCHMARK_CONFIG_DICT = {
    'env_id': "InvertedPendulum-v2",
    'MLP_V': {
        'mlp_config': [
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "1",
                "N_UNITS": 40,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.5
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "2",
                "N_UNITS": 14,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.158
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "3",
                "N_UNITS": 5,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.267
            },
            {
                "ACT": "IDENTITY",
                "B_INIT_VALUE": 0.0,
                "NAME": "OUPTUT",
                "N_UNITS": 1,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.447
            }
        ]
    },
    'POLICY': {
        'mlp_config': [
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "1",
                "N_UNITS": 40,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.5
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "2",
                "N_UNITS": 20,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.158
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "3",
                "N_UNITS": 10,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.224
            },
            {
                "ACT": "IDENTITY",
                "B_INIT_VALUE": 0.0,
                "NAME": "OUPTUT",
                "N_UNITS": 1,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.316
            }
        ]
    },
    'PPO': {
        'config_or_config_dict': {
            "gamma": 0.995,
            "lam": 0.98,
            "policy_train_iter": 10,
            "value_func_train_iter": 10,
            "clipping_range": (-0.2, 0.2),
            "beta": 1.0,
            "eta": 50,
            "log_var_init": -1.0,
            "kl_target": 0.003,
            "policy_lr": 0.00020125,
            "value_func_lr": 0.00378,
            "value_func_train_batch_size": 20,
            "lr_multiplier": 1.0
        }
    },
    'TrainTestFlow': {
        "TEST_SAMPLES_COUNT": 20,
        "TRAIN_SAMPLES_COUNT": 20,
        'config_or_config_dict': {
            "TEST_EVERY_SAMPLE_COUNT": 20,
            "TRAIN_EVERY_SAMPLE_COUNT": 20,
            "START_TRAIN_AFTER_SAMPLE_COUNT": 0,
            "START_TEST_AFTER_SAMPLE_COUNT": 0,
        }
    },
    'DEFAULT_EXPERIMENT_END_POINT': dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=1000,
                                         TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                         TOTAL_AGENT_UPDATE_COUNT=None),
}
