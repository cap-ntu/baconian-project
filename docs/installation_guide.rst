Installation Guide
==================

Baconian is easy to install. We offer the pip requirement file to install the required packages. Make sure
your machine has python 3.5, 3.6, or 3.7 with Ubuntu 16.04, 18.04 (recommend python 3.5 and ubuntu 16.04).


1. We recommend you to use anaconda to manage your package and environment, since installing the required packages may
overwrite some of already installed packages with a different version.

.. code-block:: bash

    source activate your_env

2. Install by pip or clone the source code and install the packages with requirement file:

.. code-block:: bash

    // pip install:
    pip install baconian
    // source code install
    git clone git@github.com:cap-ntu/baconian-project.git baconian
    cd ./baconian
    pip install pip -U
    pip install tensorflow==1.15.2 // or pip install tensorflow-gpu==1.15.2
    pip install -e .


Then you are free to go. You can either use the Baconian as a third party package and import it into your own project, or
directly modify it.

After you finish the above installation, you are able to run the following environments of Gym:

* algorithmic
* toy_text
* classic_control

3. Support for Gym full environments

If you want to use the full environments of gym, please refer to gym_ to obtain the license and library.

.. _gym: https://github.com/openai/gym#installing-everything/

Mostly you will need to install requirements for mujoco environments and box-2d environments.

4. Support for DeepMind Control Suit

For DeepMind control suit, you should install it by:

.. code-block:: bash

    pip install git+git://github.com/deepmind/dm_control.git

DeepMind control suit also relies on the mujoco engine which is the same as the mujoco-py environments in gym. They provide
similar tasks with slightly differences.

And for the different default mujoco key and mujoco binaries for mujoco-py and DeepMind control suit, please follow the
setting of mujoco-py and we will take care of the setting for DeepMind Control Suit at runtime.

5. We have implemented many examples, you may try them first at :doc:`Examples <examples>`

6. If you prefer to conduct a new experiments by yourself, you can follow the tutorial here :doc:`How to implement <implement_new_algo>`
