.. _xibs-faq-bunched-coasting-beams:

Bunched and Coasting Beams
--------------------------

.. admonition:: This section in short

    To simulate coasting beams, set `bunched=False` when calling the `.growth_rates` method and stop caring about the `bunch_length` argument. Two cases appear:

        - With `~.BjorkenMtingwaIBS` analytical expressions are adapted, leading to correct results.
        - With `~.NagaitsevIBS` an approximation is made using as bunch length :math:`C / 2 \pi` and a deviation to correct results might be observed. A warning will be logged to the user.

It is possible in ``xibs`` to obtain analytical IBS growth rates for simulations dealing with coasting beams.
The functionality is implemented in both analytical classes, `~.BjorkenMtingwaIBS` and `~.NagaitsevIBS`, though in the latter an approximation is made and resulting values should be expected to deviate from the correct ones.

The `.growth_rates` in both classes method provides a `bunched` boolean argument, which defaults to `True`, corresponding to a bunched beam case.
To adapt the growth rates calculation for a coasting beam, one simply has to set this argument to `False`:

.. code-block:: python

    # Let's assume your beam and optics parameters have been instantiated
    IBS = xibs.analytical.BjorkenMtingwaIBS(beam_params, optics)
    # IBS = xibs.analytical.NagaitsevIBS(beam_params, optics)  # alternatively

    # Getting growth rates for a bunched beam (default)
    rates_bunched = IBS.growth_rates(psx, epsy, sigma_delta, bunch_length)

    # Getting growth rates for a coasting beam
    rates_coasting = IBS.growth_rates(epsx, epsy, sigma_delta, bunch_length, bunched=False)

    # The two of course yield different values
    assert rates_bunched != rates_coasting  # this is True


Note that in both cases, the provided `bunch_length` argument is irrelevant: if using `BjorkenMtingwaIBS` the changes to analytical formulae take it out of the equation, and if using `NagaitsevIBS` it is ignored in favor of :math:`C / 2 \pi`.
