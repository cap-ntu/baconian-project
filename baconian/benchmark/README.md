### This directory contains the benchmark for the algorithms that are offered at Baconian
1. Status:

Currently, we are still working on the benchmark on multiple Gym tasks. For existed results, please check the `run_benchamr.py`

2. Usage

To reproduce the results, run following command:
```bash
python run_benchmark.py --env_id Pendulum-v0 --algo DDPG --count 10
```

A docker file is offered for installation of Baconian with Tensorflow GPU version, and mujoco. 
Use the docker with following example instruction:

```bash
// put your mujoco mjkey.txt at root directory of baconian project.
cp /path/to/mkjey.txt /path/to/baconian-project/ 
// build the container
cd /path/to/baconian-project/benchmark 
docker build -t baconian-test .
// runing the experiment, you can modify the passed in paramters 
docker run --gpus 1 baconian-test --env Pendulum-v0 --algo ddpg --count 1
// after finish, you need copy the log file back manually 
```

3. TODO

3.1 Discrete

Algorithm:
DQN, Dyna with DQN, MPC, Model-ensemble DQN

Task: 
CartPole, MountainCar, Acrobot, LunarLander

3.2 Continuous

Algorithm: 
DDPG, PPO, Dyna with DDPG, Dyna with PPO, MPC, iLQR, Model-ensemble PPO, Model-ensemble DDPG

Task: 
Pendulum, InvertedPendulumBulletEnv, HalfCheetah, InvertedDoublePendulumBulletEnv
