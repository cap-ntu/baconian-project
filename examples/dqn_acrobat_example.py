from baconian.algo.rl.model_free.dqn import DQN
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from baconian.agent.agent import Agent
from baconian.algo.rl.misc.epsilon_greedy import EpsilonGreedy
from baconian.core.experiment import Experiment
from baconian.core.pipelines.train_test_flow import TrainTestFlow
from baconian.common.util.schedules import PeriodicalEventSchedule


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
                  config_or_config_dict={
                      "TEST_SAMPLES_COUNT": 100,
                      "TRAIN_SAMPLES_COUNT": 100,
                      "TOTAL_SAMPLES_COUNT": 500
                  },
                  name=name + '_agent',
                  exploration_strategy=EpsilonGreedy(action_space=env_spec.action_space,
                                                     init_random_prob=0.5))
    experiment = Experiment(
        tuner=None,
        env=env,
        agent=agent,
        flow=TrainTestFlow(),
        name=name
    )

    experiment.run()


if __name__ == '__main__':
    from baconian.core.experiment_runner import single_exp_runner

    single_exp_runner(task_fn)
