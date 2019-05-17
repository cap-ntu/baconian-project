### The project is under active development, any functions might be changed at anytime.

# Baconian:  Boosting the model-based reinforcement learning 
[![Build Status](https://travis-ci.com/Lukeeeeee/baconian-project.svg?branch=master)](https://travis-ci.com/Lukeeeeee/baconian-project)
[![Documentation Status](https://readthedocs.org/projects/baconian-public/badge/?version=latest)](https://baconian-public.readthedocs.io/en/latest/?badge=latest)

Baconian [beˈkonin] is a toolbox for model-based reinforcement learning with user-friendly experiment setting-up, logging 
and visualization modules developed by [CAP](http://cap.scse.ntu.edu.sg/). We aim to develop a flexible, re-usable and 
modularized framework that can allow the user's to easily set-up a model-based rl experiments by reuse modules we 
offered.


### Release news:
- 2019.4.26 We just finished our demo paper to introduce Baconian: https://arxiv.org/abs/1904.10762
- 2019.4.4 Released the v0.1.4. Added benchmark results on DDPG, visualization of results will be given later. 
Fixed some bugs. 
- 2019.3.24 Released the v0.1.3. Added GP dynamics, fix some bugs.
- 2019.3.23 Released the v0.1.2.  Added linear dynamics, iLQR, LQR methods.

For previous news, please go [here](./old_news.md) 

### Documentation
Documentation is available at http://baconian-public.readthedocs.io/
### TODO and Road Map
Currently, the project is under activate development. We are working towards a stable 1.0 version. Details of the road map 
and future plans will be released as soon as possible. 

Currently we are working on
- [ ] Visualization module
- [ ] Simplify flow module
- [ ] State-of-art model-based algorithms: PILCO, GPS etc.
- [ ] Latent-space method supporting.


### Acknowledgement 
Inspired by the following open-source projects:

- garage: https://github.com/rlworkgroup/garage
- rllab: https://github.com/rll/rllab
- baselines: https://github.com/openai/baselines
- gym: https://github.com/openai/gym
- trpo: https://github.com/pat-coady/trpo

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
(linsen001@e.ntu.edu.sg) with detail information. I appreciate your help!
