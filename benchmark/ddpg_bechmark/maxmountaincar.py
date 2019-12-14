"""
Calculate the max sum reward for mountaincar experiment based across multiple trials
"""

import json

# Declare number of trials here (Example: if numTrials is 10, it will search through trial 1-10)
numTrials = 10
maxRewardArray = []

for i in range(numTrials):
    with open('./mountain_log_path' + str(i+1) + '/record/benchmark_agent/TEST/log.json', 'r') as f:
        result_dict = json.load(f)

    x = []
    for result in result_dict["sum_reward"]:
        x.append(result["log_val"]["__ndarray__"])

    print("TESTSET: " + str(i+1) + " - MAX:", max(x))
    maxRewardArray.append(max(x))

print("-----AVERAGE OF 10 Test Sets: ", max(maxRewardArray), "-------")



