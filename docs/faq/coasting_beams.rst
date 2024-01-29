.. _xibs-faq-geom-norm-emittances:

Geometric or Normalized Emittances
----------------------------------

.. admonition:: This section in short

    Both can be used, simply set the `normalized_emittances` parameter accordingly when calling functions asking for emittances.
    Where emittances are returned, the same type as the input ones can be expected.

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


.. _xibs-faq-sr-inputs:

Synchrotron Radiation Contribution to Emittances Evolutions
-----------------------------------------------------------

.. admonition:: This section in short

    The `.emittance_evolution` function of analytical classes can include the contribution of Synchrotron Radiation
    and quantum excitation if the following keyword arguments are provided: `sr_equilibrium_epsx`, `sr_equilibrium_epsy`,
    `sr_equilibrium_sigma_delta`, `sr_tau_x`, `sr_tau_y`, and `sr_tau_z`. They are described in the API reference, and
    how to obtain them is shown below.

See the `formalism dispatch`_ section below for more details.

.. code-block:: python

    # Provided 
    IBS = xibs.ibs(beam_params, optics, formalism=...)


TODO: write.