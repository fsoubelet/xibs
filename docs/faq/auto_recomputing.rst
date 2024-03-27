.. _xibs-faq-auto-recompute-rates-kicks:

Automatic Recomputation of IBS Kick Coefficients and Growth Rates
-----------------------------------------------------------------

It is possible in ``xibs`` to have the IBS kick class or analytical class automatically recompute the kick coefficients or growth rates, respectively.
This is done by giving a value to an optional argument, `auto_recompute_coefficients_percent` or `auto_recompute_rates_percent` respectively, in the relevant place.
The following sections explain how to do so for each.


.. _xibs-faq-auto-recompute-kick-coefficients:

Auto-Recomputing for IBS Kick Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: This section in short

    To let the class automatically determine when to recompute the IBS kick coefficients, set the `auto_recompute_coefficients_percent` argument to a value between 0 and 100 when instantiating the class.

It is possible to leave up to the kick class the decision of when to recompute the IBS kick coefficients, instead of doing so manually, by setting the `auto_recompute_coefficients_percent` argument to a value between 0 and 100 when instantiating the class.
In this case, after applying a kick the `xtrack.Particles`' emittances post-kick are compared to those pre-kick, and if the relative change exceeds the given percentage in any plane, an internal flag is set so that the coefficients will be automatically recomputed *before* the next kick application.

The values checked are: horizontal emittance, vertical emittance, bunch length and relative momentum spread.

For instance, if one provides `auto_recompute_coefficients_percent=5` and any of the above quantities changes by more than 5% after a kick (compared to before), the kick coefficients will be recomputed next time the `.apply_ibs_kick` method is called.
This allows not having to manually ask for the coefficients to be recomputed, and simplifies the loop seen in the example galleries, as one can observe below.

.. code-block:: python

    # Let's assume your beam and optics parameters have been instantiated
    IBS = xibs.kicks.KineticKickIBS(beam_params, optics, auto_recompute_coefficients_percent=10)
    # IBS = xibs.kicks.SimpleKickIBS(beam_params, optics, auto_recompute_coefficients_percent=10)

    # Let's assume your line and particles have been instantiated
    for turn in range(1, n_turns):
        # The following (commented out) line is not necessary anymore
        # IBS.compute_kick_coefficients(particles)
        IBS.apply_ibs_kick(particles)  # auto-recompute kick coefficients if needed
        line.track(particles, num_turns=1)


.. _xibs-faq-auto-recompute-growth-rates:

Auto-Recomputing for IBS Analytical Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: This section in short

    To let the class automatically determine when to recompute the IBS growth rates, set the `auto_recompute_rates_percent` argument to a value higher than 0 when calling the `.emittance_evolution` method.


It is possible to leave up to the analytical class the decision to recompute or not the IBS growth rates, instead of doing so manually, by setting the `auto_recompute_rates_percent` argument to a value higher than 0 when calling the `.emittance_evolution` method.
In this case, the horizontal and vertical emittances, bunch length and relative momentum spread at the next time step are computed.
A check is performed to see if the relative change of at least one of these values exceeds the given percentage compared to reference values stored at the last growth rates update.
If so, then growth rates are re-computed and the evolution to the next time step is calculated again.
No more check is performed, as the growth rates are then the most up-to-date they can possibly be.

For instance, if one provides `auto_recompute_rates_percent=5` and after the time step any of the above quantities changes by more than 5% from the reference values, the growth rates are recomputed and the evolution to the next time step is calculated once again.
This allows not having to manually ask for the growth rates to be recomputed, and simplifies the loop seen in the example galleries, as one can observe below.

.. code-block:: python

    # Let's assume your beam and optics parameters have been instantiated
    IBS = xibs.analytical.BjorkenMtingwaIBS(beam_params, optics)
    # IBS = xibs.analytical.NagaitsevIBS(beam_params, optics)

    # One should always have the initial growth rates computed
    IBS.growth_rates(eps_x, eps_y, sigma_delta, bunch_length)

    # Let's assume the current step's properties are already known 
    for sec in range(1, n_seconds):
        # The following (commented out) line is not necessary anymore
        # IBS.growth_rates(eps_x, eps_y, sigma_delta, bunch_length)
        new_epsx, new_epsy, new_sigma_delta, new_bunch_length = IBS.emittance_evolution(
            eps_x,
            eps_y,
            sigma_delta,
            bunch_length,
            dt=1,
            auto_recompute_rates_percent=5, # auto-recompute growth rates if needed
        )

One can find an example of this feature in the :ref:`dedicated example gallery <demo-analytical-auto-growth-rates>`.
