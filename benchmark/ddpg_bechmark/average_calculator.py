"""
Calculate the average sum reward for experiments based across multiple trials  for the last numSamples tests
"""

import json
import collections

# Declare the number of tests results to sample from (takes results starting from the end of experiments)
numSamples = 100
sumRewardArray = []

for i in range(10):
    with open('./pendulum_log_path_new' + str(i+1) + '/record/benchmark_agent/TEST/log.json', 'r') as f:
        result_dict = json.load(f)

    x = collections.deque(numSamples * [0], numSamples)
    for result in result_dict["sum_reward"]:
        # print(result["log_val"]["__ndarray__"])
        x.append(result["log_val"]["__ndarray__"])

    print("TESTSET: " + str(i+1) + " - AVERAGE OF LAST", numSamples, "sum rewards: ", sum(x) / numSamples)
    sumRewardArray.append(sum(x) / numSamples)

print("-----AVERAGE OF 10 Test Sets: ", sum(sumRewardArray)/10, "-------")



