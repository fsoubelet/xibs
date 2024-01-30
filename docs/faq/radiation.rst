.. _xibs-faq-sr-inputs:

Synchrotron Radiation Contribution
----------------------------------

.. admonition:: This section in short

    The `.emittance_evolution` method of analytical classes can include the contribution of Synchrotron Radiation and quantum excitation.
    It done automatically if specific keyword arguments are provided.
    They are described in the API reference and below, as well as how to obtain them.

When computing analytical emittance evolutions, in some scenarios the contribution of Synchrotron Radiation (SR) and quantum excitation might be significant.   
It is possible to include these contributions on top of the ``IBS`` ones when calling the `~xibs.analytical.AnalyticalIBS.emittance_evolution` method, by providing the following keyword arguments:

- `sr_equilibrium_epsx`: the horizontal equilibrium emittance due to SR, in [m] (or, if not [m], the same unit as *epsx* and *epsy*).
- `sr_equilibrium_epsy`: the vertical equilibrium emittance due to SR, in [m] (or, if not [m], the same unit as *epsx* and *epsy*).
- `sr_equilibrium_sigma_delta`: the equilibrium energy spread due to SR.
- `sr_tau_x`: the horizontal damping time due to SR, in [s].
- `sr_tau_y`: the vertical damping time due to SR, in [s].
- `sr_tau_z`: the longitudinal damping time due to SR, in [s].

.. note::
    It is possible to choose to provide normalized or geometric equilibrium emittances, as described in :ref:`this section <xibs-faq-geom-norm-emittances>`.
    What matters is that the same type as `epsx` and `epsy` is given, as indicated by the `normalized_emittances` boolean argument.
    In short: either provide all in geometric or all in normalized units, don't mix it up.

It is possible to obtain these parameters from either ``Xsuite`` or ``MAD-X``, as will be shown below.


Getting SR Parameters from Xsuite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All necessary parameters can be obtained in the result of a Twiss call on your `xtrack.Line`.
It is important however to have first configured the radiation model, and to make the Twiss call asking for these results (see the Xsuite user guide pages on `synchrotron radiation <https://xsuite.readthedocs.io/en/latest/synchrotron_radiation.html>`_ and `Twiss beams sizes <https://xsuite.readthedocs.io/en/latest/twiss.html#beam-sizes-from-twiss-table>`_).
See below:

.. code-block:: python

    # Let's assume this loads your line and its reference particle
    line = xt.Line.from_json("path_to_your_file.json")

    # Set the radiation mode to 'mean' and call twiss with 'eneloss_and_damping'
    # (see Xsuite user guide)
    line.configure_radiation(model="mean")
    twiss = line.twiss(eneloss_and_damping=True)

    # The damping times, in [s] are provided as:
    sr_tau_x, sr_tau_y, sr_tau_z = twiss["damping_constants_s"]

    # The normalized transverse equilibrium emittances, in [m], are provided as:
    sr_equilibrium_epsx = twiss["eq_nemitt_x"]
    sr_equilibrium_epsy = twiss["eq_nemitt_y"]

    # For the geometric ones, simply replace the n with g in the key:
    sr_equilibrium_epsx = twiss["eq_gemitt_x"]
    sr_equilibrium_epsy = twiss["eq_gemitt_y"]

    # We will need to store the equilibrium longitudinal emittance too for later
    sr_eq_zeta = twiss['eq_nemitt_zeta']  # or 'eq_gemitt_zeta' for geometric

    # The equilibrium momentum spread is not directly provided but can be obtained via
    # a method of the twiss result, using the equilibrium emittances obtained above.
    # Make sure to use the right type based on the one you retrieved previously
    beam_sizes = twiss.get_beam_covariance(
        nemitt_x=nemitt_x, nemitt_y=nemitt_y, gemitt_zeta=gemitt_zeta
    )

    # The value we want corresponds to the 'sigma_pzeta' key in this result, since in Xsuite
    # it is equivalent to 'sigma_delta' (see Xsuite physics guide, Eq 1.14 and 1.23). Take it
    # at the location of the particles:
    sr_equilibrium_sigma_delta = beam_sizes["sigma_pzeta"][0]  # 0 for end / start of line


Getting SR Parameters from MAD-X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While a bit more cumbersome, it is possible to get these parameters from ``MAD-X`` as well.
Let's assume your sequence and beam are defined, one might get the necessary parameters as follows:

.. code-block:: python

    # Make sure to include radiation effects for the active beam
    madx.input("bean, radiate;")

    # Let's then call the 'emit' command with DELTAP=0, which will update
    # the beam with equilibrium values directly
    madx.input("emit, deltap=0;")

    # The normalized transverse equilibrium emittances, in [m], are provided as:
    madx.input("eq_exn = beam->exn;")
    madx.input("eq_eyn = beam->eyn;")
    sr_equilibrium_epsx = madx.globals["eq_exn"]
    sr_equilibrium_epsx = madx.globals["eq_eyn"]

    # For the geometric ones, simply remove the n in the beam attribute:
    madx.input("eq_ex = beam->ex;")
    madx.input("eq_ey = beam->ey;")
    sr_equilibrium_epsx = madx.globals["eq_ex"]
    sr_equilibrium_epsx = madx.globals["eq_ey"]

    # The equilibrium momentum spread is not directly provided but can be obtained from
    # the relative energy spread using the relativistic beta as:
    madx.input("eq_sigd = beam->sige / beam->beta / beam->beta;")
    sr_equilibrium_sigma_delta = madx.globals["eq_sigd"]

    # We will need to get from the active beam: particle energy, energy loss per
    # turn (in [GeV]) and the revolution frequency (in [MHz])
    madx.input("E0 = beam->energy;")
    madx.input("U0 = beam->U0;")
    madx.input("frev = beam->freq0;")
    E0 = madx.globals["E0"] * 1e9
    U0 = madx.globals["U0"] * 1e9
    frev = madx.globals["frev"] * 1e6

    # We will need the synchrotron radiation integrals to determine the
    # damping partition numbers (see https://arxiv.org/pdf/1507.02213.pdf)
    madx.command.twiss(chrom=True)  # chrom to trigger their calculation
    I2 = madx.table.summ.synch_2[0]
    I4 = madx.table.summ.synch_4[0]
    jx = 1 - I4 / I2  # horizontal damping partition number
    jz = 2 + I4 / I2  # longitudinal damping partition number

    # This is enough to compute the damping times (see https://arxiv.org/pdf/1507.02213.pdf)
    sr_tau_x = 2 * E0 * frev / (jx * U0)
    sr_tau_y = 2 * E0 * frev / U0
    sr_tau_z = 2 * E0 * frev / (jz * U0)
