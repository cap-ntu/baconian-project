# ModelBasedRLFramework
[![Build Status](https://travis-ci.com/Lukeeeeee/ModelBasedRLFramework.svg?token=dTo6wB1jmzxu58xyRPX6&branch=master)](https://travis-ci.com/Lukeeeeee/ModelBasedRLFramework)

#### Development Path deadline: 2.19
```
1. policy, value function and algo module: 1.15
2. env, dynamics model based env : 1.19
3. player, train pipeline: 1.26
4. logging, config, unittest: 1.31
5. experiment monitor, coordinator 2. 7
6. document, test and code clean up: 2.15
```


#### Question to be considered:
```


// 2. The pipeline can be considered as a work flow or state transition model 
// 1. How to define the minimal elements that do not need to be decoupled
// in algorithm
```

#### Feature that will include:
```
// 1. global random seed control: done
// 2. unit test
// 3. error and exception handling
4. more user-friendly configuration
5. more user-friendly logging
~~6. paralleled pipeline~~
7. Support for third party envrionment simulation by network communication
8. add a debug mode for fast debugging
9. consider a new module to handle the log file content, may avoid the json dumps issue
10. paramters adaptive strategy
```

### some third party package/tools may be used
```
// 1. pytest for unit test
// 2. travis for CI
3. COVERALLS: for test coverage 
// 4. Sphinx for documentation
5. concurrent for paralleded computing?
// 6. overide for assure the siguature of functions
``` 

### TODO
```
1. algorithms that will be include: 
    model-free: dqn, ddpg, trpo, ppo, acktrs, a2c, a3c, ddqn, double dqn
    model-based: mpc, LQR, iLQR
2. Benchmark test
3. Suvery on different usage of model-based task and algorithms
4. different dynamics model form: nn, gaussian mixture model ...
```


### Old code problem:
```
1. can't detect the configuration fault.
2. take too much time to set up the experiments, 
freeze the whole configure instantly after running: how about snapshot all code?

3. consider some parser argument of script 
4. no unit test
5. configuration is too complicated
6. log files are too heavy 
7. Not to use global cfg to control the code behaviour
```

### Principle 
```
1. Get a overrall system first and optimize later
2. On the top level, users APIs  should be simply enough: only algo, agent and env module should be used.

```

### Survey on model-based rl algo

```
1. Model predictive control: approximate the dynamics with a neural network, linear system etc.
2. Embede to control, E2C:  a model that is has locally linear transition dynamics in a latent space
3. Model Ensemble Trust Region Policy Optimization (ME-TRPO): use ensemble dynamics models to better drive the learning 
of TRPO
4. 
```