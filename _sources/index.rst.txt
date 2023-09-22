Welcome to xibs' documentation!
=====================================

``xibs`` is a Python library prototype for the Intra-Beam Scattering (IBS) process modeling.

It provides, through its various modules, functionality to compute relevant IBS quantities such as growth rates, damping times, emittances evolutions etc., through different formalism.

.. admonition:: **Package Scope**

   This package only has as a goal to be a development prototype for an IBS implementation that will be integrated into `xsuite <https://xsuite.readthedocs.io/en/latest/>`_
   As such, the API is meant to evolve quickly with feedback from colleagues, and integration into ``xsuite`` simulations will not be seamless.


Installation
============

This package is tested for and supports `Python 3.8+`.
You can install it simply from with `pip`, from ``PyPI``, in a virtual environment with:

.. code-block:: bash

   python -m pip install xibs

.. tip::
    Don't know what a virtual environment is or how to set it up?
    Here is a good primer on `virtual environments <https://realpython.com/python-virtual-environments-a-primer/>`_ by `RealPython`.


Contents
========

.. toctree::
   :maxdepth: 2

   quickstart
   modules/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
