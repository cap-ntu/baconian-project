How to implement a new dynamics model
=============================================


New dynamics model in Baconian project are supposed to implement the methods and attributes defined in
``DynamicsModel`` class (``baconian/algo/dynamics/dynamics_model.py``).

.. literalinclude:: ../baconian/algo/dynamics/dynamics_model.py
    :linenos:
    :language: python
    :lines: 19-158


Similar to algorithms, dynamics models are categorized in ``baconian/algo/dynamics/dynamics_model.py``,
such as ``GlobalDynamicsModel`` and ``DifferentiableDynamics``.
