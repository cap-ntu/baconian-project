"""
This script show the example for adding an early stopping feature so when the agent can't increase its received average
reward for evaluation, the experiment will end early.

To do so in a extensible and modular way. We can implement a new flow called EarlyStoppingFlow that implement a special
ending condition detections by accessing the agent's evaluation reward (with built-in modular to access). Such mechanism
can be re-used by all algorithms, which avoid the redundant coding for users.
"""
from baconian.config.dict_config import DictConfig
from baconian.core.flow.train_test_flow import TrainTestFlow

from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.algo.ddpg import DDPG
from baconian.algo.policy import DeterministicMLPPolicy
from baconian.core.agent import Agent
from baconian.algo.misc import EpsilonGreedy
from baconian.core.experiment import Experiment
from baconian.core.status import get_global_status_collect
from baconian.common.schedules import PeriodicalEventSchedule


class EarlyStoppingFlow(TrainTestFlow):
    required_key_dict = {
        **TrainTestFlow.required_key_dict,
        'USE_LAST_K_EVALUATION_REWARD': 10
    }

    def __init__(self, train_sample_count_func, config_or_config_dict: (DictConfig, dict), func_dict: dict, agent):
        super().__init__(train_sample_count_func, config_or_config_dict, func_dict)
        self.agent = agent

    def _is_ended(self):
        test_reward = sorted(self.agent.recorder.get_log(attr_name='sum_reward', filter_by_status=dict(status='TEST')),
                             key=lambda x: x['sample_counter'])
        if len(test_reward) >= self.parameters('USE_LAST_K_EVALUATION_REWARD') * 2:
            last_reward = test_reward[-self.parameters('USE_LAST_K_EVALUATION_REWARD'):]
            pre_reward = test_reward[-self.parameters('USE_LAST_K_EVALUATION_REWARD') * 2: -self.parameters(
                'USE_LAST_K_EVALUATION_REWARD')]
            last_reward = np.mean([r['log_val'] for r in last_reward])
            pre_reward = np.mean([r['log_val'] for r in pre_reward])
            if last_reward < pre_reward:
                ConsoleLogger().print('info', 'training ended because last {} step reward: {} < previous {} step reward {}'.format(self.parameters('USE_LAST_K_EVALUATION_REWARD'), last_reward, self.parameters('USE_LAST_K_EVALUATION_REWARD'), pre_reward))
                return True
        return super()._is_ended()


def create_early_stopping_flow(test_every_sample_count, train_every_sample_count, start_train_after_sample_count,
                               start_test_after_sample_count, train_func_and_args, test_func_and_args,
                               sample_func_and_args,
                               agent,
                               use_last_k_evaluation_reward,
                               train_samples_counter_func=None):
    config_dict = dict(
        TEST_EVERY_SAMPLE_COUNT=test_every_sample_count,
        TRAIN_EVERY_SAMPLE_COUNT=train_every_sample_count,
        START_TRAIN_AFTER_SAMPLE_COUNT=start_train_after_sample_count,
        START_TEST_AFTER_SAMPLE_COUNT=start_test_after_sample_count,
        USE_LAST_K_EVALUATION_REWARD=use_last_k_evaluation_reward
    )

    def return_func_dict(s_dict):
        return dict(func=s_dict[0],
                    args=s_dict[1],
                    kwargs=s_dict[2])

    func_dict = dict(
        train=return_func_dict(train_func_and_args),
        test=return_func_dict(test_func_and_args),
        sample=return_func_dict(sample_func_and_args),
    )
    if train_samples_counter_func is None:
        def default_train_samples_counter_func():
            return get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT')

        train_samples_counter_func = default_train_samples_counter_func

    return EarlyStoppingFlow(config_or_config_dict=config_dict,
                             train_sample_count_func=train_samples_counter_func,
                             agent=agent,
                             func_dict=func_dict)


def task_fn():
    env = make('Pendulum-v0')
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
    policy = DeterministicMLPPolicy(env_spec=env_spec,
                                    name_scope=name + '_mlp_policy',
                                    name=name + '_mlp_policy',
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
                                            "N_UNITS": env_spec.flat_action_dim,
                                            "TYPE": "DENSE",
                                            "W_NORMAL_STDDEV": 0.03
                                        }
                                    ],
                                    reuse=False)

    ddpg = DDPG(
        env_spec=env_spec,
        config_or_config_dict={
            "REPLAY_BUFFER_SIZE": 10000,
            "GAMMA": 0.999,
            "CRITIC_LEARNING_RATE": 0.001,
            "ACTOR_LEARNING_RATE": 0.001,
            "DECAY": 0.5,
            "BATCH_SIZE": 50,
            "TRAIN_ITERATION": 1,
            "critic_clip_norm": 0.1,
            "actor_clip_norm": 0.1,
        },
        value_func=mlp_q,
        policy=policy,
        name=name + '_ddpg',
        replay_buffer=None
    )
    agent = Agent(env=env, env_spec=env_spec,
                  algo=ddpg,
                  algo_saving_scheduler=PeriodicalEventSchedule(
                      t_fn=lambda: get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                      trigger_every_step=20,
                      after_t=10),
                  name=name + '_agent',
                  exploration_strategy=EpsilonGreedy(action_space=env_spec.action_space,
                                                     init_random_prob=0.5))

    flow = create_early_stopping_flow(
        agent=agent,
        use_last_k_evaluation_reward=5,
        test_every_sample_count=10,
        train_every_sample_count=10,
        start_test_after_sample_count=5,
        start_train_after_sample_count=5,
        train_func_and_args=(agent.train, (), dict()),
        test_func_and_args=(agent.test, (), dict(sample_count=1)),
        sample_func_and_args=(agent.sample, (), dict(sample_count=100,
                                                     env=agent.env))
    )

    experiment = Experiment(
        tuner=None,
        env=env,
        agent=agent,
        flow=flow,
        name=name
    )
    experiment.run()


from baconian.core.experiment_runner import *

GlobalConfig().set('DEFAULT_LOG_PATH', './log_path')
GlobalConfig().set('DEFAULT_EXPERIMENT_END_POINT',
                   dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=2000,
                        TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                        TOTAL_AGENT_UPDATE_COUNT=None))
single_exp_runner(task_fn, del_if_log_path_existed=True)
