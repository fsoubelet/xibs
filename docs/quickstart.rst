Quickstart
==========

This page provides a quick overview of the package, with more usage detail given in the examples.
Please note that the package is a prototype and the API is subject to change, though this documentation will be kept up-to-date.

.. hint::

   You can click the function names in the code examples below to go directly to their documentation.

Basic Usage
-----------

One can use the package simply by importing ``xibs``:

.. code-block:: python

   import xibs

This namespace exposes the main components / classes of ``xibs``, which will be expanded on below.
Various submodules give access to different functionalities calculating relevant IBS properties through different formalism, which one can import on a per-need basis:

.. code-block:: python

   # Main IBS functionality (sub)modules
   import xibs.analytical
   import xibs.kicks.kinetic
   import xibs.kicks.simple

   # Other (sub)modules
   import xibs.inputs
   import xibs.formulary


Formalism and Models
--------------------

Here will be an overview of the different formalism and models available in the package.
For analytical calculations, one would use the ``xibs.analytical`` submodule and the provided `NagaitsevIBS` class:

.. autolink-preface:: import xibs
.. code-block:: python

   # Start with examples importing each
   pass

.. todo::

   Go over here the basic philosophy of the package: no matter the formalism used the API will work similarly.
   In time, a unified interface (e.g. `xibs.ibs(..., formalism="...")`)?


Using with xsuite
-----------------

.. todo::
   
   An overview of how to integrate this into ``xsuite``.
   While this package is used it is a prototype and integration will not be seamless.
