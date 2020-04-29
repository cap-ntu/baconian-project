from baconian.benchmark.ddpg_benchmark import mountiancar_task_fn, pendulum_task_fn
from baconian.benchmark.dyna_benchmark import dyna_pendulum_task_fn
from baconian.benchmark.mpc_benchmark import mpc_pendulum_task_fn
from baconian.benchmark.ppo_benchmark import pendulum_task_fn as ppo_pendulum_task_fn
from baconian.benchmark.ppo_benchmark import reacher_task_fn
from baconian.benchmark.ppo_benchmark import swimmer_task_fn
from baconian.benchmark.ppo_benchmark import hopper_task_fn
from baconian.benchmark.ppo_benchmark import half_cheetah_task_fn
from baconian.benchmark.ppo_benchmark import inverted_pendulum_task_fn, half_cheetah_bullet_env_task_fn
from baconian.benchmark.iLQR_benchmark import ilqr_pendulum_task_fn
from baconian.benchmark.dqn_benchmark import acrobot_task_fn, lunarlander_task_fn
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
    'HalfCheetahBulletEnv-v0': {
        'ppo': half_cheetah_bullet_env_task_fn,

    },
    'Acrobot-v1': {
        'dqn': acrobot_task_fn,
    },
    'LunarLander-v2': {
        'dqn': lunarlander_task_fn,
    }
}
alog_list = ['ddpg', 'dyna', 'mpc', 'ppo', 'ilqr', 'dqn']

arg.add_argument('--env_id', type=str, choices=list(env_id_to_task_fn.keys()))
arg.add_argument('--algo', type=str, choices=alog_list)
arg.add_argument('--count', type=int, default=1)
arg.add_argument('--cuda_id', type=int, default=-1)
args = arg.parse_args()


if __name__ == '__main__':
    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

    GlobalConfig().set('DEFAULT_LOG_PATH', os.path.join(CURRENT_PATH, 'benchmark_log', args.env_id, args.algo,
                                                        time.strftime("%Y-%m-%d_%H-%M-%S")))
    ExpRootPath = GlobalConfig().DEFAULT_LOG_PATH
    duplicate_exp_runner(args.count, env_id_to_task_fn[args.env_id][args.algo], gpu_id=args.cuda_id)
