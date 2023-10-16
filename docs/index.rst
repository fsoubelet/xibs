Welcome to xibs' documentation!
=====================================

``xibs`` is a Python library prototype for the Intra-Beam Scattering (IBS) process modeling.

It provides, through its various modules, functionality to compute relevant IBS quantities such as growth rates, damping times, emittances evolutions etc., through different formalism.

.. admonition:: **Package Scope**

   This package only has as a goal to be a development prototype for an IBS implementation that will be integrated into `xsuite <https://xsuite.readthedocs.io/en/latest/>`_.
   As such, the API is subject to quick changes based on feedback from colleagues, and integration into ``xsuite`` simulations will not be as seamless as other elements.

.. dropdown:: Useful Quick Links
    :animate: fade-in-slide-down
    :title: text-center

    .. panels::
        :card: + intro-card text-center
        :body: text-center

        ---
        :img-top: _static/index_getting_started.png

        Getting Started
        ^^^^^^^^^^^^^^^

        Check out the quickstart guide, an introduction to the package's main contents and concepts.

        +++

        .. link-button:: quickstart
            :type: ref
            :text: Quickstart
            :classes: btn-outline-primary btn-block stretched-link

        ---
        :img-top: _static/index_gallery.png

        Examples
        ^^^^^^^^

        Access various tutorials showcasing the capabilities of the package, including plots.

        +++

        .. link-button:: gallery/index
            :type: ref
            :text: Gallery
            :classes: btn-outline-primary btn-block stretched-link

        ---
        :img-top: _static/index_api.png

        API Reference
        ^^^^^^^^^^^^^

        A detailed description of how the methods work and which parameters can be used.

        +++

        .. link-button:: modules/index
            :type: ref
            :text: Reference
            :classes: btn-outline-primary btn-block stretched-link

        ---
        :img-top: _static/index_bibliography.png

        Bibliography
        ^^^^^^^^^^^^

        A compilation of the various papers are referenced throughout this documentation.

        +++

        .. link-button:: bibliography
            :type: ref
            :text: Bibliography
            :classes: btn-outline-primary btn-block stretched-link


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
   :maxdepth: 1

   quickstart
   gallery/index
   modules/index
   bibliography


Acknowledgments
===============

The following people have contributed to the development of this package by contributing code, documentation, benchmarks, comments and/or ideas:

* :user:`Felix Soubelet <fsoubelet>`
* :user:`Michalis Zampetakis <MichZampetakis>`
* :user:`Elias Waagaard <ewaagaard>`


License
=======

The package is licensed under the `Apache 2.0 license <https://github.com/fsoubelet/xibs/blob/master/LICENSE>`_. 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
