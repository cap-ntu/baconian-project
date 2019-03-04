### The project is under active development, any APIs can be changed at anytime.

# Baconian:  Boosting the model-based reinforcement learning 
 
[![Build Status](https://travis-ci.com/Lukeeeeee/baconian-internal.svg?token=dTo6wB1jmzxu58xyRPX6&branch=master)](https://travis-ci.com/Lukeeeeee/baconian-internal)
[![Documentation Status](https://readthedocs.org/projects/baconian/badge/?version=latest)](https://baconian.readthedocs.io/en/latest/?badge=latest)

Baconian [beˈkonin] is a toolbox for model-based reinforcement learning with user-friendly experiment setting-up, logging 
and visualization modules developed by [CAP](http://cap.scse.ntu.edu.sg/). We aim to develop a flexible, re-usable and 
modularized framework that can allow the user's to easily set-up a model-based rl experiments by reuse modules we 
offered.

![CAP](https://user-images.githubusercontent.com/9161548/40165577-eff023c4-59ee-11e8-8bf5-508325a23baa.png)

### Release news:
- 2019.03.04 Release the v0.1 within CAP group
### Documentation
Documentation is available at http://baconian.readthedocs.io/ (release soon, the configuration is undergoing)
#### How to install
(`source activate you_env` if you are using anaconda or consider creating a new env by `conda env create -name some_env`)
```
pip install -r requirment.txt
```
Then you are free to go. If you want to use the full environments of gym, e.g., Mujoco, please refer to its  project 
page to install the requirements (you may need to re-install gym after that.)

#### Examples
There are some examples placed at `examples` which you can have a test. 


### Todo

- [ ] Visualization module
- [ ] Optimal control algorithms: LQR, Ilqr
- [ ] State-of-art model-based algorithms
- [ ] Benchmark tests on multiple tasks

### Acknowledgement 
Thanks to the following open-source projects:

- garage: https://github.com/rlworkgroup/garage
- rllab: https://github.com/rll/rllab
- baselines: https://github.com/openai/baselines
- gym: https://github.com/openai/gym
- trpo: https://github.com/pat-coady/trpo
