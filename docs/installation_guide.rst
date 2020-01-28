Installation Guide
==================

Baconian is easy to install. We offer the pip requirement file to install required packages. Make sure
your machine have python 3.5, 3.6, or 3.7 with Ubuntu 16.04, 18.04 (recommend python 3.5 and ubuntu 16.04).


1. We recommend you to use anaconda to manage your package and environment, since installing the required packages may
overwrites some of already installed packages with a different version.

.. code-block:: bash

    source activate your_env

2. Clone the source code and install the packages with requirement file:

.. code-block:: bash

    git clone git@github.com:Lukeeeeee/baconian-project.git baconian
    cd ./baconian
    ./installation.sh

Then you are free to go. You can either use the Baconian as a third party package and import into your own project, or
directly modify it.

After you finish the above installation, you are able to run the following environments of Gym:

* algorithmic
* toy_text
* classic_control

3. Support for Gym full environments

If you want to use the full environments of gym, please refer to `gym`_ to obtain the license and library.
.. _gym: <https://github.com/openai/gym#installing-everything

Mostly you will need to install requirements for mujoco environments and box-2d environments.

4. Support for DeepMind Control Suit

For DeepMind control suit, you should install it by:

.. code-block:: bash

    pip install git+git://github.com/deepmind/dm_control.git

DeepMind control suit also rely on the mujoco engine which is the same as the mujoco-py environments in gym. They provides
similar tasks with slightly differences.

And for the different default mujoco key and mujoco binaries for mujoco-py and DeepMind control suit, please follow the
setting of mujoco-py and we will take care of the setting for DeepMind Control Suit at runtime.

4. We have implemented many examples, you may try them first at :doc:`Examples <examples>`

5. If you prefer to conduct a new experiments by yourself, you can follow the tutorial here :doc:`How to implement <implement_new_algo>`
