Logging in Baconian
=====================
In this part, we will introduce how to record and save everything you want during an experiments. The contents are
organized as follows:

* How the logging module of Baconian works.
* How to modify your new implemented module so Baconian can capture the logging information you want to save.

How the logging module of Baconian works
----------------------------------------

There are two important modules of Baconian: ``Logger`` and ``Recorder``, ``Recorder`` is coupled with every module or
class you want to record something during training or testing, for such as DQN, Agent or Environment. It will record the
information like loss, gradient or reward in a way that you specified. While ``Logger`` will take charge of these
recorded information, group them in a certain way and output them into file, console etc.