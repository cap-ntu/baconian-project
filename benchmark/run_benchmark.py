from benchmark.ddpg_bechmark import mountiancar_task_fn, pendulum_task_fn
from benchmark.dyna_benchmark import dyna_pendulum_task_fn
from benchmark.mpc_benchmark import mpc_pendulum_task_fn
from benchmark.ppo_benchmark import pendulum_task_fn as ppo_pendulum_task_fn
from benchmark.ppo_benchmark import inverted_pendulum_task_fn
from benchmark.iLQR_benchmark import ilqr_pendulum_task_fn
import argparse
import os
import time
from baconian.config.global_config import GlobalConfig
from baconian.core.experiment_runner import duplicate_exp_runner
from baconian.common.log_data_loader import MultipleExpLogDataLoader

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
    }
}
alog_list = ['ddpg', 'dyna', 'mpc', 'ppo', 'ilqr']

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
