# ModelBasedRLFramework
[![Build Status](https://travis-ci.com/Lukeeeeee/ModelBasedRLFramework.svg?token=dTo6wB1jmzxu58xyRPX6&branch=master)](https://travis-ci.com/Lukeeeeee/ModelBasedRLFramework)

#### Question to be consider:
```
1. How to define the minimal elements that do not need to be decoupled
in algorithm
2.
```

#### Feature that will include:
```
1. global random seed control: done
2. unit test
3. error and exception handling
4. more user-friendly configuration
5. more user-friendly logging
6. paralleled pipeline
7. Support for third party envrionment simulation by network communication
```

### some third party package/tools may be used
```
1. pytest for unit test
2. travis for CI
3. COVERALLS: for test coverage 
4. Sphinx for documentation
5. concurrent for paralleded computing?
6. overide for assure the siguature of functions
``` 

### TODO
```
1. algorithms that will be include: 
    dqn, ddpg, trpo, ppo, mpc, acktrs, a2c, a3c
2. Benchmark test
3. Suvery on different usage of model-based task and algorithms
```


### Old code problem:
```
1. can't detect the configuration fault.
2. take too much time to set up the experiments, 
froze the whole configure instantly after running: how about snapshot all code?

3. consider some parser argument of script 
4. no unit test
5. configuration is too complicated
```