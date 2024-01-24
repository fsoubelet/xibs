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

   # Main IBS functionality modules
   import xibs.analytical
   import xibs.kicks

   # Some other modules
   import xibs.inputs
   import xibs.formulary

In the main namespace is a convenience dispatch function to initialize the proper modelling class based on the demanded formalism.
See the `formalism dispatch`_ section below for more details.

Integration with xsuite
-----------------------

The ``xibs`` package is meant to integrate with ``xsuite`` simulations.
As a first step, all classes encompassing IBS functionality are initialized from the optics of the `xtrack.Line` to simulate for, as well as the `xpart.Particles` distribution to be tracked through the line.

.. hint::
   
   Please note that while tracking is not necessary to calculate IBS effects (see the Analytical section below), it is necessary to provide an `xpart.Particles` object from which to get required properties.
   The object does not necessarily need to represent a full generated and matched distribution, see the :doc:`FAQ <faq>` for details.

Initializing then requires the following steps:

.. code-block:: python

   import xibs

   # Let's assume your `line` and `particles` are already defined - from xsuite
   optics_parameters = xibs.inputs.OpticsParameters(line.twiss(particle_ref=p0))
   beam_parameters = xibs.inputs.BeamParameters(particles)

   # Let's say you want a YourChosenFormalismIBS approach
   IBS = YourChosenFormalismIBS(beam_parameters, optics_parameters)

   # Now compute IBS growth rates (and then updated emittances, etc.)
   IBS.growth_rates(...)

.. note::
   It is also possible to initialize the `OpticsParameters` from a ``MAD-X`` twiss table (in the form of a dataframe), see the :doc:`FAQ <faq>` for details.

Formalism and Models
--------------------

The ``xibs`` package provides functionality to calculate and apply IBS effects through different formalism.

Analytical Calculations
^^^^^^^^^^^^^^^^^^^^^^^

If one decides to stick to analytical calculations of growth rates and emittance evolutions, this is where the integration ends.
In ``xibs`` these are calculated following either:

   - The ``Nagaitsev`` formalism :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`, which provides equations for faster computation of the approach by Bjorken and Mtingwa :cite:`CERN:Bjorken_Mtingwa:Intrabeam_Scattering`.
   - The ``Bjorken-Mtingwa`` formalism :cite:`CERN:Antoniou:Revision_IBS_MADX`, which adapts the approach of Bjorken and Mtingwa :cite:`CERN:Bjorken_Mtingwa:Intrabeam_Scattering` by taking into consideration the effects of vertical dispersion.

All functionality is provided in the ``xibs.analytical`` submodule through the `NagaitsevIBS` and `BjorkenMtingwaIBS` classes, respectively.
They are initialized as shown in the section above:

.. code-block:: python

   from xibs.analytical import BjorkenMtingwaIBS, NagaitsevIBS
   from xibs.inputs import BeamParameters, OpticsParameters

   # Let's assume your `line` and `particles` are already defined
   optics_parameters = OpticsParameters(line.twiss(particle_ref=p0))
   beam_parameters = BeamParameters(particles)

   # To get analytical modelling with Nagaitsev' approach
   IBS = NagaitsevIBS(beam_parameters, optics_parameters)
   
   # To get analytical modelling with MAD-X' approach
   IBS = BjorkenMtingwaIBS(beam_parameters, optics_parameters)
   
   # Now compute IBS growth rates (and then updated emittances, etc.)
   IBS.growth_rates(...)

One can find detailed usage walkthroughs of the `BjorkenMtingwaIBS` and `NagaitsevIBS` classes usage in the :ref:`Bjorken-Mtingwa <demo-analytical-bjorken-mtingwa>` and :ref:`Nagaitsev <demo-analytical-nagaitsev>` analytical examples, respectively.

Providing Kicks to Particle Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to integrate IBS effects into tracking simulations however, computing IBS kicks to apply to the tracked particles is necessary.
For this, the ``xibs.kicks`` module is provided, which includes two submodules: `xibs.kicks.simple` and `xibs.kicks.kinetic`.

The former provides a simple kick calculation according to :cite:`PRAB:Bruce:Simple_IBS_Kicks`, which builds on the analytical formalism values from :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation` and is valid *above transition energy*.
The latter provides kicks according to the Kinetic theory of :cite:`NuclInstr:Zenkevich:Kinetic_IBS`.

Both follow the same usage pattern as the analytical formalism, and are initialized as shown in the section above:

.. code-block:: python

   from xibs.inputs import BeamParameters, OpticsParameters
   from xibs.kicks import KineticKickIBS, SimpleKickIBS

   # Let's assume your `line` and `particles` are already defined
   optics_parameters = OpticsParameters(line.twiss(particle_ref=p0))
   beam_parameters = BeamParameters(particles)

   # Initialize your class
   kinetic_ibs = KineticKickIBS(beam_parameters, optics_parameters)
   simple_ibs = SimpleKickIBS(beam_parameters, optics_parameters)
   
   # Now compute kicks to apply to particles
   simple_ibs.compute_kick_coefficients(particles)
   simple_ibs.apply_ibs_kick(particles)

One can find a detailed usage walkthrough of these in the :ref:`kinetic example <demo-kinetic-kicks>` and :ref:`simple example <demo-simple-kicks>`.

Formalism Dispatch
^^^^^^^^^^^^^^^^^^

The ``xibs`` package provides a convenience functionto initialize the proper modelling class based on the demanded formalism.
One can directly import it from the main namespace, and provide both the necessary `BeamParameters` and `OpticsParameters` to any IBS class in this package, as well as the formalism to use:

.. code-block:: python

   import xibs

   # Here is where you would define your inputs
   # Let's assume your `line` and `particles` are already defined
   beam_parameters = xibs.inputs.BeamParameters(particles)
   optics_parameters = xibs.inputs.OpticsParameters(line.twiss(particle_ref=p0))

   # Get the proper modelling class based on the demanded formalism
   BM_IBS = xibs.ibs(beam_parameters, optics_parameters, formalism="madx")
   NAGAITSEV_IBS = xibs.ibs(beam_parameters, optics_parameters, formalism="nagaitsev")
   KINETIC_IBS = xibs.ibs(beam_parameters, optics_parameters, formalism="kinetic")
   SIMPLE_IBS = xibs.ibs(beam_parameters, optics_parameters, formalism="simple")

   # You can be sure you will get the appropriate instances
   isinstance(BM_IBS, xibs.analytical.BjorkenMtingwaIBS)  # True
   isinstance(NAGAITSEV_IBS, xibs.analytical.NagaitsevIBS)  # True
   isinstance(KINETIC_IBS, xibs.kicks.KineticKickIBS)  # True
   isinstance(SIMPLE_IBS, xibs.kicks.SimpleKickIBS)  # True

   # Now go and do your IBS calculations :)
