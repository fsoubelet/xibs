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
Various submodules give access to different functionalities calculating relevant **IBS** properties through different formalism, which one can import on a per-need basis:

.. code-block:: python

   # Main IBS functionality (sub)modules
   import xibs.analytical
   import xibs.kicks.kinetic
   import xibs.kicks.simple

   # Other (sub)modules
   import xibs.inputs
   import xibs.formulary

Integration with xsuite
-----------------------

The ``xibs`` package is meant to integrate with ``xsuite`` simulations.
As a first step, all classes encompassing **IBS** functionality are initialized from the optics of the `xtrack.Line` to simulate for, as well as the `xpart.Particles` distribution to be tracked through the line.

.. tip::
   
   Please note that while tracking is not necessary to calculate **IBS** effects (see the Analytical section below), it is necessary to provide an `xpart.Particles` distribution from which to get required properties.

Initializing then requires the following steps:

.. code-block:: python

   import xibs

   # Let's assume your `line` and `particles` are already defined
   optics_parameters = xibs.inputs.OpticsParameters(line.twiss(particle_ref=p0))
   beam_parameters = xibs.inputs.BeamParameters(particles)

   # Let's say you want a YourChosenFormalismIBS approach
   IBS = YourChosenFormalismIBS(beam_parameters, optics_parameters)
   # now do some IBS calculations

Formalism and Models
--------------------

The ``xibs`` package provides functionality to calculate and apply **IBS** effects through different formalism.

Analytical Emittance Evolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If one decides to stick to analytical calculations of growth rates and emittance evolutions, this is where the integration ends.
In ``xibs`` these are calculated following Nagaitsev's approach :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`, which provides equations for faster computation of the approach by Bjorken and Mtingwa :cite:`CERN:Bjorken_Mtingwa:Intrabeam_Scattering`.

All functionality is provided in the ``xibs.analytical`` submodule through the `NagaitsevIBS` class.
It is initialized as shown in the section above:

.. code-block:: python

   from xibs.analytical import NagaitsevIBS

   # Let's assume your `line` and `particles` are already defined
   optics_parameters = xibs.inputs.OpticsParameters(line.twiss(particle_ref=p0))
   beam_parameters = xibs.inputs.BeamParameters(particles)

   # Let's say you want a YourChosenFormalismIBS approach
   IBS = NagaitsevIBS(beam_parameters, optics_parameters)
   # now compute IBS growth rates and new emittances

One can find a detailed usage walkthrough of this in the :ref:`analytical example <demo-analytical>`.
.. TODO: Change the link above to the example page once it has been built.

Providing Kicks
^^^^^^^^^^^^^^^

In order to integrate **IBS** effects into tracking simulations however, computing **IBS** kicks to apply to the tracked particles is necessary.
For this, the ``xibs.kicks`` module is provided, which includes two submodules: `xibs.kicks.simple` and `xibs.kicks.kinetic`.

The former provides a simple kick calculation according to :cite:`PRAB:Bruce:Simple_IBS_Kicks`, which valid above transition energy, and builds on the analytical formalism values from :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.
The latter provides kicks according to the Kinetic theory of :cite:`NuclInstr:Zenkevich:Kinetic_IBS`, which is valid below transition energy.

Both follow the same usage pattern as the analytical formalism, and are initialized as shown in the section above:

.. code-block:: python

   from xibs.kicks.kinetic import KineticKickIBS
   from xibs.kicks.simple import SimpleKickIBS

   # Let's assume your `line` and `particles` are already defined
   optics_parameters = xibs.inputs.OpticsParameters(line.twiss(particle_ref=p0))
   beam_parameters = xibs.inputs.BeamParameters(particles)

   # Initialize your class
   kinetic_ibs = KineticKickIBS(beam_parameters, optics_parameters)
   simple_ibs = SimpleKickIBS(beam_parameters, optics_parameters)
   # now compute kicks to apply to particles

One can find a detailed usage walkthrough of these in the :ref:`kinetic example <xibs-kinetic>` and :ref:`simple example <xibs-simple>`.
.. TODO: Change the links above to the example pages once they have been built.

.. todo::

   In time, reach: no matter the formalism used the API will work similarly?
   Aka a unified interface (e.g. `xibs.ibs(..., formalism="...")`)?
