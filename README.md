# Baconian:  Boosting the model-based reinforcement learning 
[![Build Status](https://travis-ci.com/Lukeeeeee/baconian-project.svg?branch=master)](https://travis-ci.com/Lukeeeeee/baconian-project)
[![Documentation Status](https://readthedocs.org/projects/baconian-public/badge/?version=latest)](https://baconian-public.readthedocs.io/en/latest/?badge=latest)
![GitHub issues](https://img.shields.io/github/issues/Lukeeeeee/baconian-project.svg)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/ea83a8fef57b4d8f8c9c2590337c8bc1)](https://www.codacy.com/app/Lukeeeeee/baconian?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Lukeeeeee/baconian&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/Lukeeeeee/baconian-project/branch/master/graph/badge.svg)](https://codecov.io/gh/Lukeeeeee/baconian-project)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/lukeeeeee/baconian-project.svg)
![GitHub](https://img.shields.io/github/license/Lukeeeeee/baconian-project.svg)

Baconian [beˈkonin] is a toolbox for model-based reinforcement learning with user-friendly experiment setting-up, logging 
and visualization modules developed by [CAP](http://cap.scse.ntu.edu.sg/). We aim to develop a flexible, re-usable and 
modularized framework that can allow the user's to easily set-up a model-based rl experiments by reuse modules we 
offered.


### Release news:
- 2019.10.7 Currently we are working on the benchmark test, stay tuned!
- 2019.7.30 Release the v0.2.0, updated full API documentations, added visualization module.
- 2019.6.23 Released the v0.1.5, added data pre-processing module, added support for more Gym environment (Roboschool, Atari) and deep mind control suit.


For previous news, please go [here](./old_news.md) 

### Documentation
We support python 3.5, 3.6, and 3.7 with Ubuntu 16.04 or 18.04.
Documentation is available at http://baconian-public.readthedocs.io/

### Benchmark 
Currently we are working on the benchmark test, stay tuned!

### Algorithms:

#### Model-based: 

#### 1. Dyna
Sutton, Richard S. "Dyna, an integrated architecture for learning, planning, and reacting." ACM Sigart Bulletin 2.4 (1991): 160-163.
#### 2. LQR
Abbeel, P. "Optimal Control for Linear Dynamical Systems and Quadratic Cost (‘LQR’)." (2012).
#### 3. iLQR
Abbeel, P. "Optimal Control for Linear Dynamical Systems and Quadratic Cost (‘LQR’)." (2012).
#### 4. MPC
Garcia, Carlos E., David M. Prett, and Manfred Morari. "Model predictive control: theory and practice—a survey." Automatica 25.3 (1989): 335-348.
#### Model-free

#### 1. DQN
Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
#### 2. PPO
Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
#### 3. DDPG
Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).

### Algorithms in Progress
#### 1. ME-TRPO
Kurutach, Thanard, et al. "Model-ensemble trust-region policy optimization." arXiv preprint arXiv:1802.10592 (2018).
#### 2. GPS
Levine, Sergey, et al. "End-to-end training of deep visuomotor policies." The Journal of Machine Learning Research 17.1 (2016): 1334-1373.
#### 3. Random Shooting
Rao, Anil V. "A survey of numerical methods for optimal control." Advances in the Astronautical Sciences 135.1 (2009): 497-528.
#### 4. TRPO
Schulman, John, et al. "Trust region policy optimization." International conference on machine learning. 2015.
#### 5. MB-MF
Nagabandi, Anusha, et al. "Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning." 2018 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2018.

### Acknowledgement 
Thanks to the following open-source projects:

- garage: https://github.com/rlworkgroup/garage
- rllab: https://github.com/rll/rllab
- baselines: https://github.com/openai/baselines
- gym: https://github.com/openai/gym
- trpo: https://github.com/pat-coady/trpo
- GPS: http://rll.berkeley.edu/gps/

### Citing Baconian
If you find Baconian is useful for your research, please consider cite our demo paper here:
```
@article{
linsen2019baconian, 
title={Baconian: A Unified Opensource Framework for Model-Based Reinforcement Learning}, 
author={Linsen, Dong and Guanyu, Gao and Yuanlong, Li and Yonggang, Wen}, 
journal={arXiv preprint arXiv:1904.10762},
year={2019} 
}
```
### Report an issue 
If you find any bugs on issues during your usage of the package, please open an issue or send an email to me 
(linsen001@e.ntu.edu.sg) with detailed information. I appreciate your help!
