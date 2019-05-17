Tutorial (Highly recommend to read before you start)
========================================================

Here we introduce some very basic usage about the Baconian, to make sure you utilize the code correctly. As
for detail usage of different algorithms, dynamics, please refer to the :doc:`API <API>` page

Put everything into a function
------------------------------
Here we introduce the basic usage of baconian, and introduce how it can help you
to set up the model-based RL experiments.

First of all, whenever you want to run some algorithms, or any codes within the
baconian, simply you need to wrap your code into a function, and pass this
function to ``single_exp_runner`` or ``duplicate_exp_runner``. In this method, Baconian will do some internal initialization of logging, experiment
set-up etc.

``single_exp_runner`` will run your function for once. As for ``duplicate_exp_runner``, it is designed for running
multiple experiments in a row, because in RL experiments, we usually run the experiment with a certain set of parameters but with different seeds to get a more
stable results. So use ``duplicate_exp_runner`` can easily help you to achieve this, and the log file
will be stored into sub-directory under your home log directory respectively.

Specifically, you can do it by:

.. code-block:: python

    from baconian.core.experiment_runner import single_exp_runner, duplicate_exp_runner
    # Define your function first.
    def you_function():
        a = 1;
        b = 2;
        print(a + b)
    # Then pass the function object to single_exp_runner, then it will set up everything and run your code.
    single_exp_runner(you_function)
    # Or call duplicate_exp_runner to run multiple experiments in a row. 10 is the number of experiments:
    duplicate_exp_runner(10, you_function)


Global Configuration Usage
---------------------------
The global configuration offer the setting including default log path, log level, and some other system related default
configuration. We implement the global configuration module with singleton method, and you can utilize it by following
examples:

.. code-block:: python

    from baconian.config.global_config import GlobalConfig
    from baconian.core.experiment_runner import single_exp_runner, duplicate_exp_runner
    def you_function():
        a = 1;
        b = 2;
        print(a + b)
    # Use GlobalConfig() to access the instance of GlobalConfig
    # anywhere your want, and set the log path by yourself
    # First argument is key you want to set, e.g., DEFAULT_LOG_PATH
    # Second argument is the value.
    GlobalConfig().set('DEFAULT_LOG_PATH', './log_path')
    single_exp_runner(task_fn, del_if_log_path_existed=True)


During the time task is running, the global configuration will be frozen, if you try to change it, an error will be
raised.

Train and Test Workflow for RL Experiments
--------------------------------------------
In Baconian, the control flow of the experiments is delegated to an independent module ``baconian.core.flow.train_test_flow:Flow``
which is  an abstract class. The reason to do so is to improve the flexibility and extensibility of framework.
Two typical flow are implemented. One is
``baconian.core.flow.train_test_flow:TrainTestFlow``, which corresponds to most of
model-free algorithms pipeline, which is sampling-training-testing pipeline. Another one is
``baconian.core.flow.dyna_flow.py:DynaFlow``, which is the flow in Dyna algorithms [Sutton, 1992].

.. note::
    Do get freaked out by the long list of parameters you need to initialize the flow!
    This is caused by the fact that we want the flow to be fully configurable by users, currently we are working to simplify this module.

Work with Status Control
-------------------------
Status control is a must for DRL experiments. For instance, off-policy DRL methods need to switch between behavior
policy and target policy during sampling and testing or decay the exploration action noise w.r.t the training progress status.

Every class that inherited from ``baconian.core.core:Basic`` will have two class attributes: ``STATUS_LIST`` which contains
all status of this class or module, and ``INIT_STATUS`` as the initial status of it.

You can call ``set_status`` method to change the status of one instance. You can call ``get_status`` method to get the
current status of an instance, which is a dict type. The return value not only contains the status (i.e., TRAIN, TEST)
but also other extra information that is specially added in the code. Such as, you can register a counter of a function
by doing so:

.. literalinclude:: ../baconian/core/agent.py
    :linenos:
    :language: python
    :lines: 121-136

The counter for calling function ``predict`` is added as one attribute of status, which will be returned with the key
``predict_counter``

For detailed usage of these methods, you can find it in API reference.
