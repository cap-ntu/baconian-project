from benchmark.ddpg_bechmark import mountiancar_task_fn, pendulum_task_fn
import argparse
import os
import time
from baconian.config.global_config import GlobalConfig
from baconian.core.experiment_runner import duplicate_exp_runner

arg = argparse.ArgumentParser()
env_id_to_task_fn = {
    'Pendulum-v0': pendulum_task_fn,
    'MountainCarContinuous-v0': mountiancar_task_fn
}

arg.add_argument('--env_id', type=str, choices=list(env_id_to_task_fn.keys()))
args = arg.parse_args()

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

GlobalConfig.set('DEFAULT_LOG_PATH', os.path.join(CURRENT_PATH, 'benchmark_log', args.env_id,
                                                  time.strftime("%Y-%m-%d_%H-%M-%S")))
duplicate_exp_runner(10, env_id_to_task_fn[args.env_id])
