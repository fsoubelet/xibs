"""

.. _demo-compare-analytical:

==================================================================
Comparison of Analytical IBS Growth Rates and Emittances Evolution
==================================================================

This example shows a comparison of the performance of `~xibs.analytical.BjorkenMtingwaIBS`
and `~xibs.analytical.NagaitsevIBS` vs ``MAD-X`` when computing emittances evolutions from
analytical IBS growth rates.

We will do the comparison by using the case of the CERN LHC, with protons at top
energy (aka 6 800 GeV at the moment), including the crossing schemes, which introduce
vertical dispersion through the machine.

.. note::
    This example requires the `acc-models-lhc` repository next to this file, which
    you can get with the following command:

    .. code-block:: bash
    
        git clone -b 2023 https://gitlab.cern.ch/acc-models/acc-models-lhc.git --depth 1

"""
# sphinx_gallery_thumbnail_number = 2
import logging
import warnings

from dataclasses import dataclass
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from helpers import _get_dummy_ibs_from_madx_rates, prepare_all

warnings.simplefilter("ignore")  # for this tutorial's clarity
logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] [%(levelname)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 15,
        "figure.titlesize": 20,
    }
)

###############################################################################
# Let's start by setting up the lattice in ``MAD-X`` (we are using `cpymad` for
# this) and our IBS classes. The following helper functions calls a configuration
# file from this repository's tests, sets up the lattice and beam in ``MAD-X`` and
# creates corresponding `~xibs.analytical.BjorkenMtingwaIBS` and
# `~xibs.analytical.NagaitsevIBS`.

params, madx, BM_IBS, NAG_IBS = prepare_all(
    configname="lhc_top_protons",  # fetches the config file with this name
    extrafile="acc-models-lhc/strengths/ATS_Nominal/2023/ats_30cm.madx",  # to have xing
    stdout=False,  # don't want the MAD-X output
)

###############################################################################
# Here `params` hold the properties we want to be analytically tracking.
# We can check that crossing angles are present in the ``MAD-X`` lattice, and
# lead to vertical dispersion:

pprint(params)
print(f"Crossing angles in IP1/5: {madx.globals['on_x1']}, {madx.globals['on_x5']} urad")

table = madx.table.twiss.dframe()
plt.figure(figsize=(11, 6))
plt.plot(table.s, table.dx, lw=2, label="Vertical")
plt.plot(table.s, table.dy, lw=2, label="Horizontal")
plt.ylabel(r"$D_{x,y}$ [m]")
plt.xlabel("Longitudinal location [m]")
plt.legend()
plt.tight_layout()
plt.show()

###############################################################################
# Comparing Analytical Evolution for a Time Period
# ------------------------------------------------
# We will analytically look at the evolution through time by looping, just as in
# the :ref:`Bjorken-Mtingwa <demo-analytical-bjorken-mtingwa>` and
# :ref:`Nagaitsev <demo-analytical-nagaitsev>` analytical examples, respectively.
# Let's do so for 10 hours of beam time, recomputing the growth rates every 30 minutes.

# Duration to track for and frequency at which to re-compute the growth rates, both in [s]
nsecs = 10 * 3_600  # that's 10h
ibs_step = 30 * 60  # re-compute rates every 30min
seconds = np.linspace(0, nsecs, nsecs).astype(int)
beta_rel: float = BM_IBS.beam_parameters.beta_rel


# Set up a dataclass to store the results
@dataclass
class Records:
    """Dataclass to store (and update) important values through tracking."""

    eps_x: np.ndarray  # geometric horizontal emittance in [m]
    eps_y: np.ndarray  # geometric vertical emittance in [m]
    sigd: np.ndarray  # momentum spread
    bl: np.ndarray  # bunch length in [m]


# Initialize dataclasses for each approach
madx_tbt = Records(
    np.zeros(nsecs, dtype=float),
    np.zeros(nsecs, dtype=float),
    np.zeros(nsecs, dtype=float),
    np.zeros(nsecs, dtype=float),
)
bm_tbt = Records(
    np.zeros(nsecs, dtype=float),
    np.zeros(nsecs, dtype=float),
    np.zeros(nsecs, dtype=float),
    np.zeros(nsecs, dtype=float),
)
nag_tbt = Records(
    np.zeros(nsecs, dtype=float),
    np.zeros(nsecs, dtype=float),
    np.zeros(nsecs, dtype=float),
    np.zeros(nsecs, dtype=float),
)

# Initialize data at second 0 (initial state) for all structures
madx_tbt.eps_x[0] = bm_tbt.eps_x[0] = nag_tbt.eps_x[0] = params.geom_epsx
madx_tbt.eps_y[0] = bm_tbt.eps_y[0] = nag_tbt.eps_y[0] = params.geom_epsy
madx_tbt.sigd[0] = bm_tbt.sigd[0] = nag_tbt.sigd[0] = params.sig_delta
madx_tbt.bl[0] = bm_tbt.bl[0] = nag_tbt.bl[0] = params.bunch_length

#############################################################################
# With the settings above and this loop, we compute the emittances
# every second but we only update the growth rates every 30 minutes.

for sec in range(1, nsecs):
    # ----- Potentially re-compute the IBS growth rates ----- #
    if (sec % ibs_step == 0) or (sec == 1):
        print(f"At {sec}s: re-computing growth rates")
        # For MAD-X - call the 'twiss' then 'ibs' commands (computes from beam attributes)
        MAD_IBS = _get_dummy_ibs_from_madx_rates(
            madx
        )  # calls twiss, ibs, gets rates and puts them in returned 'dummy IBS'
        MAD_IBS.beam_parameters = BM_IBS.beam_parameters  # we will want to access these
        MAD_IBS.optics = BM_IBS.optics  # we will want to access these
        # For Bjorken-Mtingwa - compute from values at the previous sec
        BM_IBS.growth_rates(
            bm_tbt.eps_x[sec - 1],
            bm_tbt.eps_y[sec - 1],
            bm_tbt.sigd[sec - 1],
            bm_tbt.bl[sec - 1],
        )
        # For Nagaitsev - compute from values at the previous sec
        NAG_IBS.growth_rates(
            nag_tbt.eps_x[sec - 1],
            nag_tbt.eps_y[sec - 1],
            nag_tbt.sigd[sec - 1],
            nag_tbt.bl[sec - 1],
        )

    # ----- Compute the new parameters using current growth rates ----- #
    # For MAD-X - compute from values at previous second and specify dt=1s
    madx_epsx, madx_epsy, madx_sigd, madx_bl = MAD_IBS.emittance_evolution(
        madx_tbt.eps_x[sec - 1],
        madx_tbt.eps_y[sec - 1],
        madx_tbt.sigd[sec - 1],
        madx_tbt.bl[sec - 1],
        dt=1,
    )
    # For Bjorken-Mtingwa - compute from values at previous second and specify dt=1s
    bm_epsx, bm_epsy, bm_sigd, bm_bl = BM_IBS.emittance_evolution(
        bm_tbt.eps_x[sec - 1],
        bm_tbt.eps_y[sec - 1],
        bm_tbt.sigd[sec - 1],
        bm_tbt.bl[sec - 1],
        dt=1,
    )
    # For Nagaitsev - compute from values at previous second and specify dt=1s
    nag_epsx, nag_epsy, nag_sigd, nag_bl = NAG_IBS.emittance_evolution(
        nag_tbt.eps_x[sec - 1],
        nag_tbt.eps_y[sec - 1],
        nag_tbt.sigd[sec - 1],
        nag_tbt.bl[sec - 1],
        dt=1,
    )

    # ----- Update the records with the new values ----- #
    # For MAD-X
    madx_tbt.eps_x[sec] = madx_epsx
    madx_tbt.eps_y[sec] = madx_epsy
    madx_tbt.sigd[sec] = madx_sigd
    madx_tbt.bl[sec] = madx_bl

    # For Bjorken-Mtingwa
    bm_tbt.eps_x[sec] = bm_epsx
    bm_tbt.eps_y[sec] = bm_epsy
    bm_tbt.sigd[sec] = bm_sigd
    bm_tbt.bl[sec] = bm_bl

    # For Nagaitsev
    nag_tbt.eps_x[sec] = nag_epsx
    nag_tbt.eps_y[sec] = nag_epsy
    nag_tbt.sigd[sec] = nag_sigd
    nag_tbt.bl[sec] = nag_bl

    # ----- Update the beam attributes for MAD-X ----- #
    madx.sequence.lhcb1.beam.ex = madx_epsx
    madx.sequence.lhcb1.beam.ey = madx_epsy
    madx.sequence.lhcb1.beam.sige = madx_sigd * beta_rel**2
    madx.sequence.lhcb1.beam.sigt = madx_bl

###############################################################################
# Feel free to run this simulation for more turns, or with a different frequency
# of the IBS growth rates re-computation. After this is done running, we can plot
# the evolutions across the turns:

fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(13, 7))

# Plotting horizontal emittances
axs["epsx"].plot(seconds / 3600, 1e10 * madx_tbt.eps_x, lw=2.5, label="MAD-X")
axs["epsx"].plot(seconds / 3600, 1e10 * bm_tbt.eps_x, lw=1.5, label="BjorkenMtingwaIBS")
axs["epsx"].plot(seconds / 3600, 1e10 * nag_tbt.eps_x, lw=1, label="NagaitsevIBS")

# Plotting vertical emittances
axs["epsy"].plot(seconds / 3600, 1e10 * madx_tbt.eps_y, lw=2.5, label="MAD-X")
axs["epsy"].plot(seconds / 3600, 1e10 * bm_tbt.eps_y, lw=1.5, label="BjorkenMtingwaIBS")
axs["epsy"].plot(seconds / 3600, 1e10 * nag_tbt.eps_y, lw=1.5, label="NagaitsevIBS")

# Plotting momentum spread
axs["sigd"].plot(seconds / 3600, 1e4 * madx_tbt.sigd, lw=2.5, label="MAD-X")
axs["sigd"].plot(seconds / 3600, 1e4 * bm_tbt.sigd, lw=1.5, label="BjorkenMtingwaIBS")
axs["sigd"].plot(seconds / 3600, 1e4 * nag_tbt.sigd, lw=1, label="NagaitsevIBS")

# Plotting bunch length
axs["bl"].plot(seconds / 3600, 1e2 * madx_tbt.bl, lw=2.5, label="MAD-X")
axs["bl"].plot(seconds / 3600, 1e2 * bm_tbt.bl, lw=1.5, label="BjorkenMtingwaIBS")
axs["bl"].plot(seconds / 3600, 1e2 * nag_tbt.bl, lw=1, label="NagaitsevIBS")

# Axes parameters
axs["epsx"].set_ylabel(r"$\varepsilon_x$ [$10^{-10}$m]")
axs["epsy"].set_ylabel(r"$\varepsilon_y$ [$10^{-10}$m]")
axs["sigd"].set_ylabel(r"$\sigma_{\delta}$ [$10^{-4}$]")
axs["bl"].set_ylabel(r"Bunch length [cm]")
axs["epsx"].legend()

for axis in (axs["epsy"], axs["bl"]):
    axis.yaxis.set_label_position("right")
    axis.yaxis.tick_right()

for axis in (axs["sigd"], axs["bl"]):
    axis.set_xlabel("Duration [h]")

for axis in axs.values():
    axis.xaxis.set_major_locator(plt.MaxNLocator(8))

fig.align_ylabels((axs["epsx"], axs["sigd"]))
fig.align_ylabels((axs["epsy"], axs["bl"]))

# Figure parameters
fig.suptitle("LHC Top Protons w/ Xing")
plt.tight_layout()
plt.show()

#############################################################################
# Notice how we observe a great agreement between different implementations,
# except for the vertical emittances. This is expected, as the lattice setup
# includes vertical dispersion which is not taken into consideration by the
# `~.xibs.analytical.NagaitsevIBS` formalism.

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `~xibs.analytical`: `~.xibs.analytical.BjorkenMtingwaIBS`, `~.xibs.analytical.NagaitsevIBS`, `~.xibs.analytical.BjorkenMtingwaIBS.growth_rates`, `~.xibs.analytical.BjorkenMtingwaIBS.emittance_evolution`

###############################################################################
