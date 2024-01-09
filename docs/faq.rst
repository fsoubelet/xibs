.. _xibs-faq:

Frequently Asked Questions
==========================

This page provides answers to common questions or pitfalls about the package.

.. hint::

   If you feel like a topic should be included here, let us know!


.. _xibs-faq-opticsparams:

Instantiating OpticsParameters
------------------------------

All classes in this package are instantiated from an `OpticsParameters` and a `BeamsParameters` object.
This section compiles a few common questions about the former.


.. _xibs-faq-optics-params-from-line:

Instantiating OpticsParameters from an `xtrack.Line`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default mode to instantiate an `OpticsParameters` is from the `TwissTable` returned by an `xtrack.Line.twiss` call.
All necessary attributes of the class are automatically filled or computed from the `TwissTable` object. 
It goes very simply as:

.. code-block:: python

    # Let's assume your `xtrack.Line` is already defined
    twiss = line.twiss(particle_ref=p0)
    optics_params = OpticsParameters(twiss)

.. note::
    Since ``xibs`` version 0.3.0, there is a convenience method to do the above automatically for the user.
    It is documented in the :ref:`API reference <xibs-inputs>`.
    It goes according to:

    .. code-block:: python

        # Let's assume your `xtrack.Line` is already defined
        optics_params = OpticsParameters.from_line(line)


.. _xibs-faq-optics-params-from-madx:

Instantiating OpticsParameters from ``MAD-X``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For those using for instance ``MAD-X`` and looking to only use the analytical calculations of ``xibs`` without setting up a whole `xtrack.Line`, it is possible to instantiate the `OpticsParameters` from a ``MAD-X`` `twiss` result.

The `twiss` result is expected in the form of a `pandas.DataFrame` with all lowercase columns and values, as given by `cpymad`.
Some additional attributes not present in ``MAD-X``'s `twiss` result are expected to be provided at instantiation time, namely the revolution frequency and slip factor. 
These quantities can be computed from ``MAD-X`` after a `twiss`, as shown below:

.. code-block:: python

    # Let's assume your `cpymad.madx.Madx` instance is already defined
    twiss = madx.twiss(centre=True).dframe()  # you want a Twiss at element centers
    seq_name = madx.table.twiss.summary.sequence  # works whichever your sequence

    # Compute additional required parameters: revolution frequency and slip factor
    frev_hz = madx.sequence[seq_name].beam.freq0 * 1e6  # beware freq0 is in MHz
    gamma_rel = madx.sequence[seq_name].beam.gamma  # relativistic gamma
    gamma_tr = madx.table.summ.gammatr[0]  # transition gamma
    slipfactor = (1/(gamma_tr**2)) - (1/(gamma_rel**2))  # use the xsuite convention!

    # And these have to be provided as additional arguments
    optics_params = OpticsParameters(twiss, slipfactor, frev_hz)

.. note::
    Since ``xibs`` version 0.3.0, there is a convenience method to do the above automatically for the user.
    It is documented in the :ref:`API reference <xibs-inputs>`.

    .. code-block:: python

        # Let's assume your `cpymad.madx.Madx` instance is already defined
        optics_params = OpticsParameters.from_madx(madx)


.. _xibs-faq-beamparams:

Instantiating BeamParameters
----------------------------

All classes in this package are instantiated from an `OpticsParameters` and a `BeamsParameters` object.
This section compiles a few common questions about the latter.


.. _xibs-faq-beam-params-from-particle-ref:

Do I need a full matched particle distribution for `BeamParameters`?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One might have noticed, for example in the :ref:`Bjorken-Mtingwa <demo-analytical-bjorken-mtingwa>` or :ref:`Nagaitsev <demo-analytical-nagaitsev>` analytical examples, that the latter is instantiated from a fully matched `xpart.Particles` distribution.
Creating a matched distribution is an intensive step and requires parameters one might not want to be annoyed providing.

It is possible to bypass this step and use for instance the `xtrack.Line`'s `.particle_ref` instead.
However, in several computations the Coulomb logarithm is necessary, which is heavily dependent on the number of particles in the distribution, a value not reflected by the reference particle object.

When instantiating from a single particle this value will be wrong and should be manually set afterwards.
This would go as:

.. code-block:: python

    import xpart as xp
    from xibs.inputs import BeamParameters

    # Let's define what one would use as "reference particle" for the line and use that
    p0 = xp.Particles(p0c=6500e9, q0=1, mass0=xp.PROTON_MASS_EV)
    beam_params = BeamParameters(p0)

    # Now you have to manually set the "true" value for '.n_part'
    beam_parameters.n_part = int(5e5)


.. _xibs-faq-beam-params-from-line:

Instantiating BeamParameters from an `xtrack.Line`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default mode to instantiate a `BeamParameters` is from an `xpart.Particles` object.
As seen just above, it is possible to use an `xtrack.Line`'s reference particle to do so.


Since ``xibs`` version 0.3.0, there is a convenience method to do the above automatically for the user.
It is documented in the :ref:`API reference <xibs-inputs>`.
It goes according to:

.. code-block:: python

    # Let's assume your `xtrack.Line` is already defined
    beam_params = BeamParameters.from_line(line, n_part=5e5)  # need to provide n_part


.. _xibs-faq-beam-params-from-madx:

Instantiating BeamParameters from ``MAD-X``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to query the `beam` in use for the currently active sequence from ``MAD-X`` to get the desired parameters.


.. code-block:: python

    # Let's assume your `cpymad.madx.Madx` instance is already defined
    madx.command.twiss()  # want the table to determine the sequence name and access its beam
    seq_name = madx.table.twiss.summary.sequence  # will give us the active sequence

    # Query required parameters from the beam: particle momentum, particle charge,
    # particle rest mass and number of particles in the bunch 
    p0c_eV = madx.sequence[seq_name].beam.pc * 1e9  # in [GeV] in MAD-X beam, but we want [eV]
    q0 = madx.sequence[seq_name].beam.charge  # electrical particle charge in units of [qp]
    mass0 = madx.sequence[seq_name].beam.mass * 1e9  # rest mass in [eV] | but in [GeV] in MAD-X
    npart = madx.sequence[seq_name].beam.npart  # number of particles

    # Create an xpart.Particles object with this information
    particle = xp.Particles(p0c=p0c_eV, q0=q0, mass0=mass0)
    beam_params = BeamParameters(particle)
    beam_params.n_part = int(npart)  # very important to adjust this!

.. note::
    Since ``xibs`` version 0.3.0, there is a convenience method to do the above automatically for the user.
    It is documented in the :ref:`API reference <xibs-inputs>`.

    .. code-block:: python

        # Let's assume your `cpymad.madx.Madx` instance is already defined
        beam_params = BeamParameters.from_madx(madx)


.. _xibs-faq-geom-norm-emittances:

Geomettric or Normalized Emittances
-----------------------------------

Some functions in ``xibs`` require emittances to be provided as input, for instance `~xibs.analytical.BjorkenMtingwaIBS.growth_rates`.
In all such cases, while internally ``xibs`` uses geomtric emittances for computations (as they are the values used in the implemented formulae), it is possible to provide either the geometric or the normalized emittances.

For these functions the API will ask for both `epsx` and `epsy` arguments, and offer an optional boolean argument `normalized_emittances` which defaults to `False`.
If set to `True`, then the provided emittances are assumed to be the normalized ones, and will be converted to geometric emittances internally.

For instance for the `~.BjorkenMtingwaIBS.growth_rates` method, the API is:

.. code-block:: python

    # Let's assume your beam and optics parameters have been instantiated
    IBS = xibs.ibs(beam_params, optics, formalism=...)
    
    # Getting growth rates from geometric emittances goes as:
    rates_geom = IBS.growth_rates(geom_epsx, geom_epsy, sigma_delta, bunch_length)

    # Getting growth rates from normalized emittances goes as:
    rates_norm = IBS.growth_rates(
        norm_epsx, norm_epsy, sigma_delta, bunch_length, normalized_emittances=True
    )

    # The two results are the same
    assert rates_geom == rates_norm  # this is True

For functions that also return emittance values, such as the `emittance_evolution` method of analytical IBS implementations, the returned values will be the same as the input ones, i.e. if normalized emittances were provided, then normalized emittances will be returned.
That's as much as the user has to think about this.


.. _xibs-faq-bunched-coasting-beams:

Bunched and Coasting Beams
--------------------------

Users may want to obtain IBS growth rates for simulations in which they have coasting beams.
This is possible, through currently only in the Bjorken & Mtingwa formalism.

.. warning::
    Keep in mind that for a coasting beam case the bunch length should be adapted, and is generally taken as the circumference of the machine.

The `~.BjorkenMtingwaIBS.growth_rates` method provides a `bunched` boolean argument, which defaults to `True`, corresponding to a bunched beam case.
To adapt the growth rates calculation for a coasting beam, one simply has to set this argument to `False`:

.. code-block:: python

    # Let's assume your beam and optics parameters have been instantiated
    IBS = xibs.analytical.BjorkenMtingwaIBS(beam_params, optics)

    # Getting growth rates for a bunched beam (default)
    rates_bunched = IBS.growth_rates(psx, epsy, sigma_delta, bunch_length)

    # Getting growth rates for a coasting beam
    rates_coasting = IBS.growth_rates(epsx, epsy, sigma_delta, bunch_length, bunched=False)

    # The two of course yield different values
    assert rates_bunched != rates_coasting  # this is True
