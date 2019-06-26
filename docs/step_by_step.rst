Step by step guide
============================================================

This is a step by step guide of how to compose an model-based RL experiments in Baconian, we will take the example of
Dyna algorithms, which is a very typical model-based RL architecture proposed by Sutton in 1990.

In this method, we need the environment and the task we aim to solve, a model-free method, an approximator as dynamics for real world
environment. We also need to specific the hyper-parameters in Dyna architecture, e.g., how much samples we should get
from real environment to update the dynamics model. how much samples from real environment and dynamics model to serve
the training of the model-free algorithms.

.. note::
    Complete codes can be found at :doc:`here <./example/dyna>`

Create the tasks and DDPG algorithms
-------------------------------------

Create env Pendulum-v0, and the DDDPG algorithms to solve the task

.. literalinclude:: ../examples/dyna.py
    :linenos:
    :language: python
    :lines: 22-91

Create a global dynamics model with MLP network
--------------------------------------------------
.. literalinclude:: ../examples/dyna.py
    :linenos:
    :language: python
    :lines: 93-121

Create the Dyna architecture as algorithms
-----------------------------
Create the Dyna algorithms by passing the ddpg and dynamics model, and wrap by the agent

.. literalinclude:: ../examples/dyna.py
    :linenos:
    :language: python
    :lines: 122-143

Configure the workflow and experiments object
---------------------
Create the dyna-like workflow, and the experiments object, and run the experiment within your `task_fn`

.. literalinclude:: ../examples/dyna.py
    :linenos:
    :language: python
    :lines: 145-178
.. note::
    Don't confuse the workflow with the dyna algorithms itself. The flow only specifics how the algorithms interact
    with environments, and how to update and evaluate the ddpg (model-free method) and dynamics model.

Set the global configuration and launch the experiment
--------------------------
Set the log path and wrap all the above task into a function `task_fn`, and pass into experiment runner.

.. literalinclude:: ../examples/dyna.py
    :linenos:
    :language: python
    :lines: 181-184

Results logging/ Visualization
--------------------------------
Please refer to :doc:`Logging and Visualization <./how_to_log>`.