PENDULUM_BENCHMARK_CONFIG_DICT = {
    'env_id': "Pendulum-v0",
    'MLP_V': {
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
                "ACT": "IDENTITY",
                "B_INIT_VALUE": 0.0,
                "NAME": "OUPTUT",
                "N_UNITS": 1,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            }
        ]
    },
    'POLICY': {
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
                "ACT": "IDENTITY",
                "B_INIT_VALUE": 0.0,
                "NAME": "OUPTUT",
                "N_UNITS": 1,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            }
        ]
    },
    'PPO': {
        'config_or_config_dict': {
            "gamma": 0.99,
            "lam": 0.95,
            "policy_train_iter": 4,
            "value_func_train_iter": 4,
            "clipping_range": [0.2, 0.2],
            "beta": 1.0,
            "eta": 50,
            "log_var_init": -1.0,
            "kl_target": 0.03,
            "policy_lr": 0.0003,
            "value_func_lr": 0.0003,
            "value_func_train_batch_size": 64,
            "lr_multiplier": 1.0
        }
    },
    'TrainTestFlow': {
        "TEST_SAMPLES_COUNT": 800,
        "TRAIN_SAMPLES_COUNT": 2048,
        'config_or_config_dict': {
            "TEST_EVERY_SAMPLE_COUNT": 1000,
            "TRAIN_EVERY_SAMPLE_COUNT": 10,
            "START_TRAIN_AFTER_SAMPLE_COUNT": 0,
            "START_TEST_AFTER_SAMPLE_COUNT": 0,
        }
    },
    'DEFAULT_EXPERIMENT_END_POINT': dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=1000000,
                                         TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                         TOTAL_AGENT_UPDATE_COUNT=None),
}
