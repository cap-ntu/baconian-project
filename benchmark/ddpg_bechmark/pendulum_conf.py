from baconian.common.noise import *
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from baconian.algo.rl.model_free.ddpg import DDPG
from baconian.algo.rl.policy.deterministic_mlp import DeterministicMLPPolicy
from baconian.core.agent import Agent
from baconian.algo.rl.misc.epsilon_greedy import EpsilonGreedy
from baconian.core.experiment import Experiment
from baconian.core.pipelines.train_test_flow import TrainTestFlow
from baconian.config.global_config import GlobalConfig
from benchmark.ddpg_bechmark.mountain_car_continuous_conf import *
from baconian.common.schedules import LinearSchedule
from baconian.core.status import get_global_status_collect
from baconian.common.noise import *

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
                "ACT": "TANH",
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
            "CRITIC_TRAIN_ITERATION": 120,
            "ACTOR_TRAIN_ITERATION": 120,
            "critic_clip_norm": 0.0,
            "actor_clip_norm": 0.0,
        },
        'replay_buffer': None
    },
    'TrainTestFlow': {
        "TEST_SAMPLES_COUNT": 2000,
        "TRAIN_SAMPLES_COUNT": 10,
        'config_or_config_dict': {
            "TEST_EVERY_SAMPLE_COUNT": 200,
            "TRAIN_EVERY_SAMPLE_COUNT": 10,
            "START_TRAIN_AFTER_SAMPLE_COUNT": 0,
            "START_TEST_AFTER_SAMPLE_COUNT": 0,
        }
    },
    'EpsilonGreedy': {
        'initial_p': 1.0,
        'final_p': 0.0,
        'schedule_timesteps': 10000
    },
    'DEFAULT_EXPERIMENT_END_POINT': dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=10000,
                                         TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                         TOTAL_AGENT_UPDATE_COUNT=None),
    'AGENT_NOISE': {

    }
}
