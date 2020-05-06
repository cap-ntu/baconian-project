from baconian.common.log_data_loader import *


if __name__ == "__main__":
    MultipleExpLogDataLoader(
        exp_root_dir_list='/Users/lukeeeeee/Code/baconian-internal/baconian/benchmark/benchmark_log/Pendulum-v0/ppo/2020-05-05_09-35-28') \
        .plot_res(sub_log_dir_name='benchmark_agent/TEST',
                  key='sum_reward',
                  index='sample_counter',
                  mode='line',
                  average_over=1,
                  file_name=None,
                  save_format='png',
                  save_path='./')
