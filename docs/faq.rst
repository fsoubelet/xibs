Frequently Asked Questions
==========================

This page provides answers to common questions or pitfalls about the package.

.. hint::

   If you feel like a topic should be included here, let us know!


Instantiating `OpticsParameters` and `BeamParameters`
-----------------------------------------------------

All classes in this package are instantiated from a `OpticsParameters` and a `BeamsParameters` object.
This section compiles a few common questions about these two classes.


Instantiating `OpticsParameters` from an `xtrack.Line.twiss` result
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default mode to instantiate an `OpticsParameters` is from the `TwissTable` returned by a `xtrack.Line.twiss` call.
All necessary attributes of the class are automatically filled or computed from the `TwissTable` object. 
It goes very simply as:

.. code-block:: python

    from xibs.inputs import OpticsParameters

    # Let's assume your `xtrack.Line` is already defined
    twiss = Line.twiss(particle_ref=p0)
    optics_params = OpticsParameters(twiss)


Instantiating `OpticsParameters` from a ``MAD-X`` `twiss` result
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For those using for instance ``MAD-X`` and looking to only use the analytical calculations of `xibs` without setting up a whole `xtrack.Line`, it is possible to instantiate the `OpticsParameters` from a ``MAD-X`` `twiss` result.

The `twiss` result is expected in the form of a `pandas.DataFrame` with all lowercase columns and values, as given by `cpymad`.
Some additional attributes not present in ``MAD-X``'s `twiss` result are expected to be provided at instantiation time. 
It goes as:

.. code-block:: python

    from xibs.inputs import OpticsParameters

    # Let's assume your `cpymad.madx.Madx` instance is already defined
    twiss = madx.twiss().dframe()
    
    # Let's define the required attributes here
    seq_name = "lhcb1"  # for instance, if we're working with LHC beam 1
    slipfactor = 0.0003484  # to be computed by the user | TODO: figure out how
    frev_hz = madx.sequence[seq_name].beam.freq0 * 1e6  # beware freq0 is in MHz

    # And these have to be provided as additional arguments
    optics_params = OpticsParameters(twiss, slipfactor, frev_hz)


Do I need a full matched particle distribution for `BeamParameters`?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One might have noticed, for example in the :ref:`analytical example <demo-analytical>`, that the latter is instantiated from a fully matched `xpart.Particles` distribution.
Creating a matched distribution is an intensive step and requires parameters one might not want to be annoyed providing.

It is possible to bypass this step and use for instance the `xtrack.Line`'s `.particle_ref` instead.
However, in several computations the Coulomb logarithm is necessary, which is heavily dependent on the number of particles in the distribution, a value not reflected by the reference particle object.

When instantiating from a single particle this values will be wrong and should be manually set afterwards.
This would go as:

.. code-block:: python

    import xpart as xp
    from xibs.inputs import BeamParameters

    # Let's define what one would use as "reference particle" for the line and use that
    p0 = xp.Particles(p0c=6500e9, q0=1, mass0=xp.PROTON_MASS_EV)
    beam_params = BeamParameters(p0)

    # Now you have to manually set the "true" value for '.n_part'
    beam_parameters.n_part = int(5e5)