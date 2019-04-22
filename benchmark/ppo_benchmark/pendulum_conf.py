PENDULUM_BENCHMARK_CONFIG_DICT = {
    'env_id': "Pendulum-v0",
    'MLP_V': {
        'mlp_config': [
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "1",
                "N_UNITS": 30,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "2",
                "N_UNITS": 40,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "3",
                "N_UNITS": 5,
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
                "N_UNITS": 30,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "2",
                "N_UNITS": 17,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "3",
                "N_UNITS": 10,
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
            "gamma": 0.9999,
            "lam": 0.98,
            "policy_train_iter": 10,
            "value_func_train_iter": 10,
            "clipping_range": None,
            "beta": 1.0,
            "eta": 50,
            "log_var_init": -1.0,
            "kl_target": 0.03,
            "policy_lr": 0.01,
            "value_func_lr": 0.01,
            "value_func_train_batch_size": 128,
            "lr_multiplier": 1.0
        }
    },
    'TrainTestFlow': {
        "TEST_SAMPLES_COUNT": 800,
        "TRAIN_SAMPLES_COUNT": 1,
        'config_or_config_dict': {
            "TEST_EVERY_SAMPLE_COUNT": 1000,
            "TRAIN_EVERY_SAMPLE_COUNT": 10,
            "START_TRAIN_AFTER_SAMPLE_COUNT": 0,
            "START_TEST_AFTER_SAMPLE_COUNT": 0,
        }
    },
    'DEFAULT_EXPERIMENT_END_POINT': dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=10000,
                                         TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                         TOTAL_AGENT_UPDATE_COUNT=None),
}
