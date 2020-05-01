"""
In this example, we demonstrate how to utilize the scheduler module to dynamically setting the
learning rate of your algorithm, or epsilon-greedy probability
"""

from baconian.algo.dqn import DQN
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.core.agent import Agent
from baconian.algo.misc import EpsilonGreedy
from baconian.core.experiment import Experiment
from baconian.core.flow.train_test_flow import create_train_test_flow
from baconian.config.global_config import GlobalConfig
from baconian.common.schedules import LinearScheduler, PiecewiseScheduler, PeriodicalEventSchedule
from baconian.core.status import get_global_status_collect


def task_fn():
    env = make('Acrobot-v1')
    name = 'example_scheduler_'
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
                                         LEARNING_RATE=0.001,
                                         TRAIN_ITERATION=1,
                                         DECAY=0.5),
              name=name + '_dqn',
              value_func=mlp_q)
    agent = Agent(env=env, env_spec=env_spec,
                  algo=dqn,
                  name=name + '_agent',
                  algo_saving_scheduler=PeriodicalEventSchedule(
                      t_fn=lambda: get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                      trigger_every_step=20,
                      after_t=10),
                  exploration_strategy=EpsilonGreedy(action_space=env_spec.action_space,
                                                     prob_scheduler=PiecewiseScheduler(
                                                         t_fn=lambda: get_global_status_collect()(
                                                             'TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                                                         endpoints=((10, 0.3), (100, 0.1), (200, 0.0)),
                                                         outside_value=0.0
                                                     ),
                                                     init_random_prob=0.5))
    flow = create_train_test_flow(
        test_every_sample_count=10,
        train_every_sample_count=10,
        start_test_after_sample_count=5,
        start_train_after_sample_count=5,
        train_func_and_args=(agent.train, (), dict()),
        test_func_and_args=(agent.test, (), dict(sample_count=10)),
        sample_func_and_args=(agent.sample, (), dict(sample_count=100,
                                                     env=agent.env,
                                                     store_flag=True))
    )
    experiment = Experiment(
        tuner=None,
        env=env,
        agent=agent,
        flow=flow,
        name=name + 'experiment_debug'
    )

    dqn.parameters.set_scheduler(param_key='LEARNING_RATE',
                                 scheduler=LinearScheduler(
                                     t_fn=experiment.TOTAL_AGENT_TRAIN_SAMPLE_COUNT,
                                     schedule_timesteps=GlobalConfig().DEFAULT_EXPERIMENT_END_POINT[
                                         'TOTAL_AGENT_TRAIN_SAMPLE_COUNT'],
                                     final_p=0.0001,
                                     initial_p=0.01))
    experiment.run()


from baconian.core.experiment_runner import single_exp_runner

GlobalConfig().set('DEFAULT_LOG_PATH', './log_path')
single_exp_runner(task_fn, del_if_log_path_existed=True)
