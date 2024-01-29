.. _xibs-faq-sr-inputs:

Synchrotron Radiation Contribution
----------------------------------

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