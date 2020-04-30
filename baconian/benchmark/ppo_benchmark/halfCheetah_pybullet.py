"""
PPO benchmark on HalfCheetahBulletEnv-v0
"""
from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.algo.value_func import MLPVValueFunc
from baconian.algo.ppo import PPO
from baconian.algo.policy.normal_distribution_mlp import NormalDistributionMLPPolicy
from baconian.core.agent import Agent
from baconian.core.experiment import Experiment
from baconian.core.flow.train_test_flow import TrainTestFlow
from baconian.config.global_config import GlobalConfig
from baconian.core.status import get_global_status_collect
from baconian.benchmark.ppo_benchmark.halfCheetah_bullet_conf import HALF_CHEETAH_BENCHMARK_CONFIG_DICT
from baconian.envs.env_wrapper import StepObservationWrapper


def half_cheetah_task_fn():
    exp_config = HALF_CHEETAH_BENCHMARK_CONFIG_DICT
    GlobalConfig().set('DEFAULT_EXPERIMENT_END_POINT',
                       exp_config['DEFAULT_EXPERIMENT_END_POINT'])

    env = make(exp_config['env_id'])

    env.reset()
    env = StepObservationWrapper(env, step_limit=env.unwrapped._max_episode_steps)
    name = 'benchmark'
    env_spec = EnvSpec(obs_space=env.observation_space,
                       action_space=env.action_space)

    mlp_v = MLPVValueFunc(env_spec=env_spec,
                          name_scope=name + 'mlp_v',
                          name=name + 'mlp_v',
                          **exp_config['MLP_V'])
    policy = NormalDistributionMLPPolicy(env_spec=env_spec,
                                         name_scope=name + 'mlp_policy',
                                         name=name + 'mlp_policy',
                                         **exp_config['POLICY'],
                                         output_low=env_spec.action_space.low,
                                         output_high=env_spec.action_space.high,
                                         reuse=False)

    ppo = PPO(
        env_spec=env_spec,
        **exp_config['PPO'],
        value_func=mlp_v,
        stochastic_policy=policy,
        name=name + '_ppo',
        use_time_index_flag=True
    )
    agent = Agent(env=env,
                  env_spec=env_spec,
                  algo=ppo,
                  exploration_strategy=None,
                  noise_adder=None,
                  name=name + '_agent')

    flow = TrainTestFlow(
        train_sample_count_func=lambda: get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_FUNC_COUNT'),
        config_or_config_dict=exp_config['TrainTestFlow']['config_or_config_dict'],
        func_dict={
            'test': {'func': agent.test,
                     'args': list(),
                     'kwargs': dict(sample_count=exp_config['TrainTestFlow']['TEST_SAMPLES_COUNT']),
                     },
            'train': {'func': agent.train,
                      'args': list(),
                      'kwargs': dict(),
                      },
            'sample': {'func': agent.sample,
                       'args': list(),
                       'kwargs': dict(sample_count=exp_config['TrainTestFlow']['TRAIN_SAMPLES_COUNT'],
                                      env=agent.env,
                                      sample_type='trajectory',
                                      in_which_status='TRAIN',
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
