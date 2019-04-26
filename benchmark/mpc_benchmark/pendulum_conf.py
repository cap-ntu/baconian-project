PENDULUM_BENCHMARK_CONFIG_DICT = {
    'env_id': "Pendulum-v0",
    'DEFAULT_EXPERIMENT_END_POINT': dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=10000,
                                         TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                         TOTAL_AGENT_UPDATE_COUNT=None),
    'DynaFlow': {
        "TEST_ALGO_EVERY_REAL_SAMPLE_COUNT": 200,
        "TEST_DYNAMICS_EVERY_REAL_SAMPLE_COUNT": 200,
        "TRAIN_DYNAMICS_EVERY_REAL_SAMPLE_COUNT": 10,
        "START_TRAIN_ALGO_AFTER_SAMPLE_COUNT": 1,
        "START_TRAIN_DYNAMICS_AFTER_SAMPLE_COUNT": 1,
        "START_TEST_ALGO_AFTER_SAMPLE_COUNT": 1,
        "START_TEST_DYNAMICS_AFTER_SAMPLE_COUNT": 1,
        "WARM_UP_DYNAMICS_SAMPLES": 2000,
        "TRAIN_ALGO_EVERY_REAL_SAMPLE_COUNT_FROM_REAL_ENV": 10,
        "TRAIN_ALGO_EVERY_REAL_SAMPLE_COUNT_FROM_DYNAMICS_ENV": 50,
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
                          ]),
    'MPC': dict(
        SAMPLED_HORIZON=5,
        SAMPLED_PATH_NUM=20,
        dynamics_model_train_iter=10
    ),
}
