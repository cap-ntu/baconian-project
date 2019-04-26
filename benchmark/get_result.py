import glob
import os
import json_tricks as json
import numpy as np


def get_reward_json(root_dir_list, sub_dir, key, index_key):
    all_res = []
    for rt in root_dir_list:
        with open(file=os.path.join(rt, sub_dir)) as f:
            res = json.load(f)[key]
            val = [rr['log_val'] for rr in res]

            index = [rr[index_key] for rr in res]
            all_res.append((val, index))
            print(val[-5:], rt)
    aver = 0.0
    for re in all_res:
        aver += float(np.mean(re[0][-5:]))
    aver /= len(all_res)
    print(aver)


if __name__ == '__main__':
    get_reward_json(
        root_dir_list=glob.glob('/home/dls/CAP/baconian-internal/benchmark/benchmark_log/Pendulum-v0/dyna/**/*'),
        sub_dir='record/benchmark_agent/TEST/log.json',
        key='sum_reward',
        index_key='predict_counter'
    )
