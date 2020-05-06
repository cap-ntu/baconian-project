Step by step guide to run a RL experiment
============================================================

This is a step by step guide of how to compose an model-based RL experiments in Baconian, we will take the example of
Dyna algorithm, which is a very typical model-based RL architecture proposed by Sutton in 1990.

In this method, we need the environment and the task we aim to solve, a model-free method, an approximator as dynamics for real world
environment. We also need to specific the hyper-parameters in Dyna architecture, e.g., how many samples we should get
from real environment to update the dynamics model, and how many samples from real environment and dynamics model to serve
the training of the model-free algorithms.

For more information on the modules used in the following codes, please see :doc:`Best Practices <./best_practice>`
and :doc:`API references <./API>`

.. note::
    Complete codes can be found at :doc:`here <./example/dyna>`

Create the tasks and DDPG algorithms
-------------------------------------

Create environment Pendulum-v0, and the DDPG algorithms to solve the task

.. literalinclude:: ../baconian/examples/dyna.py
    :linenos:
    :language: python
    :lines: 23-94

Create a global dynamics model with MLP network
--------------------------------------------------
.. literalinclude:: ../baconian/examples/dyna.py
    :linenos:
    :language: python
    :lines: 95-125

Create the Dyna architecture as algorithms
--------------------------------------------------
Create the Dyna algorithms by passing the ddpg and dynamics model, and wrap by the agent

.. literalinclude:: ../baconian/examples/dyna.py
    :linenos:
    :language: python
    :lines: 126-152

Configure the workflow and experiments object
---------------------------------------------------------------
Create the dyna-like workflow, and the experiments object, and run the experiment within your `task_fn`

.. literalinclude:: ../baconian/examples/dyna.py
    :linenos:
    :language: python
    :lines: 153-189
.. note::
    Don't confuse the workflow with the Dyna algorithm itself. The flow only specifics how the algorithms interact
    with environments, and how to update and evaluate the ddpg (model-free method) and dynamics model.

Set the global configuration and launch the experiment
--------------------------------------------------------------------
Set some global config if needed and wrap all the above task into a function `task_fn`, and pass into experiment runner.

.. literalinclude:: ../baconian/examples/dyna.py
    :linenos:
    :language: python
    :lines: 191-205

Results logging/ Visualization
--------------------------------
Please refer to :doc:`Logging and Visualization <./how_to_log>`.