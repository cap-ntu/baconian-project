from baconian.algo.rl.model_free.dqn import DQN
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from baconian.core.agent import Agent
from baconian.algo.rl.misc.epsilon_greedy import EpsilonGreedy
from baconian.core.experiment import Experiment
from baconian.core.pipelines.train_test_flow import TrainTestFlow
from baconian.config.global_config import GlobalConfig
from baconian.core.status import get_global_status_collect


def task_fn():
    env = make('Acrobot-v1')
    name = 'demo_exp'
    env_spec = EnvSpec(obs_space=env.observation_space,
                       action_space=env.action_space)

    mlp_q = MLPQValueFunction(env_spec=env_spec,
                              name_scope=name + '_mlp_q',
                              name=name + '_mlp_q',
                              mlp_config=[
                                  {
                                      "ACT": "RELU",
                                      "B_INIT_VALUE": 0.0,
                                      "NAME": "1",
                                      "N_UNITS": 16,
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
                              ])
    dqn = DQN(env_spec=env_spec,
              config_or_config_dict=dict(REPLAY_BUFFER_SIZE=1000,
                                         GAMMA=0.99,
                                         BATCH_SIZE=10,
                                         Q_NET_L1_NORM_SCALE=0.001,
                                         Q_NET_L2_NORM_SCALE=0.001,
                                         LEARNING_RATE=0.01,
                                         TRAIN_ITERATION=1,
                                         DECAY=0.5),
              name=name + '_dqn',
              value_func=mlp_q)

    agent = Agent(env=env, env_spec=env_spec,
                  algo=dqn,
                  name=name + '_agent',
                  exploration_strategy=EpsilonGreedy(action_space=env_spec.action_space,
                                                     init_random_prob=0.5))

    flow = TrainTestFlow(train_sample_count_func=lambda: get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                         config_or_config_dict={
                             "TEST_EVERY_SAMPLE_COUNT": 10,
                             "TRAIN_EVERY_SAMPLE_COUNT": 10,
                             "START_TRAIN_AFTER_SAMPLE_COUNT": 5,
                             "START_TEST_AFTER_SAMPLE_COUNT": 5,
                         },
                         func_dict={
                             'test': {'func': agent.test,
                                      'args': list(),
                                      'kwargs': dict(sample_count=10),
                                      },
                             'train': {'func': agent.train,
                                       'args': list(),
                                       'kwargs': dict(),
                                       },
                             'sample': {'func': agent.sample,
                                        'args': list(),
                                        'kwargs': dict(sample_count=100,
                                                       env=agent.env,
                                                       in_test_flag=False,
                                                       store_flag=True),
                                        },
                         })

    experiment = Experiment(
        tuner=None,
        env=env,
        agent=agent,
        flow=flow,
        name=name
    )
    experiment.run()


from baconian.core.experiment_runner import single_exp_runner

GlobalConfig.set('DEFAULT_LOG_PATH', './log_path')
single_exp_runner(task_fn)
