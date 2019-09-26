from benchmark.ddpg_bechmark import mountiancar_task_fn, pendulum_task_fn
from benchmark.dyna_benchmark import dyna_pendulum_task_fn
from benchmark.mpc_benchmark import mpc_pendulum_task_fn
from benchmark.ppo_benchmark import pendulum_task_fn as ppo_pendulum_task_fn
from benchmark.ppo_benchmark import reacher_task_fn
from benchmark.ppo_benchmark import swimmer_task_fn
from benchmark.ppo_benchmark import hopper_task_fn
from benchmark.ppo_benchmark import half_cheetah_task_fn
from benchmark.ppo_benchmark import inverted_pendulum_task_fn

from benchmark.iLQR_benchmark import ilqr_pendulum_task_fn
from examples.dqn_acrobot_example import task_fn as dqn_acrobot_task_fn
import argparse
import os
import time
from baconian.config.global_config import GlobalConfig
from baconian.core.experiment_runner import duplicate_exp_runner

arg = argparse.ArgumentParser()
env_id_to_task_fn = {
    'Pendulum-v0': {
        'ddpg': pendulum_task_fn,
        'dyna': dyna_pendulum_task_fn,
        'mpc': mpc_pendulum_task_fn,
        'ppo': ppo_pendulum_task_fn,
        'ilqr': ilqr_pendulum_task_fn
    },
    'MountainCarContinuous-v0': {
        'ddpg': mountiancar_task_fn,

    },
    'InvertedPendulum-v2': {
        'ppo': inverted_pendulum_task_fn,
    },
    'Reacher-v2': {
        'ppo': reacher_task_fn,
    },
    'Swimmer-v2': {
        'ppo': swimmer_task_fn,
    },
    'Hopper-v2': {
        'ppo': hopper_task_fn,
    },
    'HalfCheetah-v2': {
        'ppo': half_cheetah_task_fn,
    },
    'Acrobot-v1': {
        'dqn': dqn_acrobot_task_fn,
    }
}
alog_list = ['ddpg', 'dyna', 'mpc', 'ppo', 'ilqr', 'dqn']

arg.add_argument('--env_id', type=str, choices=list(env_id_to_task_fn.keys()))
arg.add_argument('--algo', type=str, choices=alog_list)
arg.add_argument('--count', type=int, default=1)
arg.add_argument('--cuda_id', type=int, default=-1)
args = arg.parse_args()

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

GlobalConfig().set('DEFAULT_LOG_PATH', os.path.join(CURRENT_PATH, 'benchmark_log', args.env_id, args.algo,
                                                    time.strftime("%Y-%m-%d_%H-%M-%S")))
ExpRootPath = GlobalConfig().DEFAULT_LOG_PATH
duplicate_exp_runner(args.count, env_id_to_task_fn[args.env_id][args.algo], gpu_id=args.cuda_id)

# MultipleExpLogDataLoader(exp_root_dir_list='/home/cly/Documents/baconian-internal/benchmark/benchmark_log/InvertedPendulum-v2/ppo/2019-09-23_11-53-12', num=args.count)\
#                          .plot_res(sub_log_dir_name='benchmark_agent/TRAIN',
#                          key='sum_reward', index='sample_counter',
#                          mode='line', average_over=1, file_name=None, save_format='png',)
