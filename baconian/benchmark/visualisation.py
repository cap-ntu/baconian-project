from baconian.common.log_data_loader import *

exp_root_dir_list = [
    '/Users/lukeeeeee/Code/baconian-internal/examples/log/',
]

if __name__ == "__main__":
    MultipleExpLogDataLoader(exp_root_dir_list=exp_root_dir_list) \
        .plot_res(sub_log_dir_name='demo_exp_agent/TRAIN',
                  key='sum_reward',
                  index='sample_counter',
                  mode='line',
                  average_over=1,
                  file_name=None,
                  save_format='png',
                  save_path='/baconian/examples/log')
