"""

.. _demo-kicks-auto-coefficients:

==========================================
Automatic Kick Coefficients Re-Computation
==========================================

This example shows how to use the the auto recomputing functionality for kick
coefficients in the kick classes. To follow this example please first have a
look at the :ref:`Kinetic Kicks example <demo-kinetic-kicks>` as well as the
relevant :ref:`FAQ section <xibs-faq-auto-recompute-kick-coefficients>`.

This script will showcase the functionality by following an identical flow to the 
Kinetic kicks example linked above and expanding on the relevant parts. Demonstration
will be done using the CLIC Damping Ring line.
"""
# sphinx_gallery_thumbnail_number = 1
# import warnings
import logging
# import sys
from dataclasses import dataclass
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xobjects as xo
import xpart as xp
import xtrack as xt

from xibs.formulary import _bunch_length, _geom_epsx, _geom_epsy, _percent_change, _sigma_delta
from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] [%(levelname)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
)
# warnings.simplefilter("ignore")  # for this tutorial's clarity
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
# Let's start by defining the line and particle information, as well as some
# parameters for later use. Note that `bunch_intensity` is the actual number
# of particles in the bunch, and its value influences the IBS kick coefficients
# calculation while `n_part` is the number of generated particles for
# tracking, which is much lower.

line_file = "lines/chrom-corr_DR.newlattice_2GHz.json"
bunch_intensity = int(4.5e9)
n_part = int(1.5e3)
sigma_z = 1.5e-3
nemitt_x = 5.5e-7
nemitt_y = 3.5e-9


###############################################################################
# Setting up line and particles
# -----------------------------
# Let's now load the line from file, activate the accelerating cavities and
# create a context for multithreading with OpenMP, since tracking particles
# is going to take some time:

line = xt.Line.from_json(line_file)
context = xo.ContextCpu(omp_num_threads="auto")
line.particle_ref = xp.Particles(mass0=xp.ELECTRON_MASS_EV, q0=1, p0c=2.86e9)

# ----- Power accelerating cavities ----- #
for cavity in [element for element in line.elements if isinstance(element, xt.Cavity)]:
    cavity.lag = 180  # we are above transition

line.build_tracker(context, extra_headers=["#define XTRACK_MULTIPOLE_NO_SYNRAD"])
line.optimize_for_tracking()
twiss = line.twiss()

particles = xp.generate_matched_gaussian_bunch(
    num_particles=n_part,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    line=line,
)
particles2 = particles.copy()


###############################################################################
# We can compute initial (geometrical) emittances as well as the bunch length
# from the `xtrack.Particles` object:

geom_epsx = _geom_epsx(particles, twiss.betx[0], twiss.dx[0])
geom_epsy = _geom_epsy(particles, twiss.bety[0], twiss.dy[0])
bunch_l = _bunch_length(particles)
sig_delta = _sigma_delta(particles)


###############################################################################
# Generating the IBS Kick Class
# -----------------------------
# Just as all user-facing classes in ``xibs``, the `KineticKickIBS` :cite:`NuclInstr:Zenkevich:Kinetic_IBS`
# class is instantiated by providing a `BeamParameters` and an `OpticsParameters` objects:

beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
opticsparams = OpticsParameters.from_line(line)

# This is the threshold that would trigger the auto-recomping of the growth rates.
AUTO_PERCENT = 8  # threshold: 8% change from kick application
IBS = KineticKickIBS(beamparams, opticsparams)
AUTO_IBS = KineticKickIBS(beamparams, opticsparams, auto_recompute_coefficients_percent=AUTO_PERCENT)


###############################################################################
# Computing and Applying IBS Kicks
# --------------------------------
# The so-called "kick coefficients" are computed directly from the particle distribution
# by calling the dedicated method. They correspond to the diffusion minus friction terms.

IBS.compute_kick_coefficients(particles2)
AUTO_IBS.compute_kick_coefficients(particles2)

###############################################################################
# Let's see the evolution of the transverse emittances, :math:`\sigma_{\delta}`
# and bunch length from a kick. We can have a look at the relative change of
# the particles's distribution properties from before to after.

IBS.apply_ibs_kick(particles2)
new_geom_epsx = _geom_epsx(particles2, twiss.betx[0], twiss.dx[0])
new_geom_epsy = _geom_epsy(particles2, twiss.bety[0], twiss.dy[0])
new_sig_delta = _sigma_delta(particles2)
new_bunch_length = _bunch_length(particles2)

# Let's see the relative change
print(f"Geom. epsx: {geom_epsx:.2e} -> {new_geom_epsx:.2e} | ({100 * _percent_change(geom_epsx, new_geom_epsx):.2e}% change)")
print(f"Geom. epsy: {geom_epsy:.2e} -> {new_geom_epsy:.2e} | ({100 * _percent_change(geom_epsy, new_geom_epsy):.2e}% change)")
print(f"Sigma delta: {sig_delta:.2e} -> {new_sig_delta:.2e} | ({100 * _percent_change(sig_delta, new_sig_delta):.2e}% change)")
print(f"Bunch length: {bunch_l:.4e} -> {new_bunch_length:.4e} | ({100 * _percent_change(bunch_l, new_bunch_length):.2e}% change)")

# Let's reset these
particles2 = particles.copy()  


###############################################################################
# Preparing for the Tracking Simulation
# -------------------------------------
# We will loop over turns for 1000 turns, and compare results from the regular
# and auto-recomputing scenarios. Let's set up the utilities needed for this:

nturns = 1000  # number of turns to loop for
ibs_step = 25  # frequency at which to re-compute coefficients in [turns]
turns = np.linspace(0, nturns, nturns, dtype=int)  # array of tracked turns


# Set up a dataclass to store the results
@dataclass
class Records:
    epsilon_x: np.ndarray
    epsilon_y: np.ndarray
    sigma_delta: np.ndarray
    bunch_length: np.ndarray

    def update_at_turn(self, turn: int, parts: xp.Particles, twiss: xt.TwissTable):
        self.epsilon_x[turn] = _geom_epsx(parts, twiss.betx[0], twiss.dx[0])
        self.epsilon_y[turn] = _geom_epsy(parts, twiss.bety[0], twiss.dy[0])
        self.sigma_delta[turn] = _sigma_delta(parts)
        self.bunch_length[turn] = _bunch_length(parts)

    @classmethod
    def init_zeroes(cls, n_turns: int) -> Self:  # noqa: F821
        return cls(
            epsilon_x=np.zeros(n_turns, dtype=float),
            epsilon_y=np.zeros(n_turns, dtype=float),
            sigma_delta=np.zeros(n_turns, dtype=float),
            bunch_length=np.zeros(n_turns, dtype=float),
        )


# Initialize the dataclass & store the initial values
regular = Records.init_zeroes(nturns)
regular.update_at_turn(0, particles, twiss)

auto = Records.init_zeroes(nturns)
auto.update_at_turn(0, particles, twiss)

# These arrays we will use to see when the auto-recomputing kicks in
auto_recomputes = np.zeros(nturns, dtype=float)
fixed_recomputes = np.zeros(nturns, dtype=float)

###############################################################################
# Tracking Evolution Over Turns
# -----------------------------
# Let's now loop and let the auto-recomputing do its job.

for turn in range(1, nturns):
    # This is not necessary, just for showcasing in this tutorial
    n_recomputes_for_auto = AUTO_IBS._number_of_coefficients_computations

    # ----- Potentially re-compute the IBS kick coefficients ----- #
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"Turn {turn:d}: Fixed interval re-computing of coefficients")
        # Below always re-computes the coefficients every 'ibs_step' turns
        IBS.compute_kick_coefficients(particles)

    # ----- Manually Apply IBS Kick and Track Turn ----- #
    IBS.apply_ibs_kick(particles)
    AUTO_IBS.apply_ibs_kick(particles2)  # auto-recomputes if necessary
    line.track(particles, num_turns=1)
    line.track(particles2, num_turns=1)

    # ----- Update records for tracked particles ----- #
    regular.update_at_turn(turn, particles, twiss)
    auto.update_at_turn(turn, particles2, twiss)

    # ----- Check if the rates were auto-recomputed ----- #
    # This is also not necessary, just for showcasing in this tutorial
    if AUTO_IBS._number_of_coefficients_computations > n_recomputes_for_auto:
        print(f"At turn {turn} - Auto re-computed coefficients")
        auto_recomputes[turn - 1] = 1


# Here we also aggregate the fixed recomputes
for turn in turns:
    if (turn % ibs_step == 0) or (turn == 1):
        fixed_recomputes[turn - 1] = 1

# And these are simply 1D arrays of the turns at which re-computing happened
where_auto_recomputes = np.flatnonzero(auto_recomputes)
where_fixed_recomputes = np.flatnonzero(fixed_recomputes)

###############################################################################
# Let's see how many recomputes happened in total for each scenario:

print(f"Fixed re-computes: {IBS._number_of_coefficients_computations}")
print(f"Auto re-computes: {AUTO_IBS._number_of_coefficients_computations}")

###############################################################################
# That's almost TODO additional updates of the coeffifients that were deemed
# necessary by the auto-recomputing mechanism. Let's see the effect it has had
# on our results by having a look at the evolution of emittances over time:

AVG_TURNS = 3  # for rolling average plotting
fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(13, 7))

# We will add vertical lines at the times where recomputing of the kick 
# coefficients happened (do this first so they show up in the background)
for axis in axs.values():
    for turn in where_fixed_recomputes:
        axis.axvline(turn, color="gray", linestyle="--", lw=1, alpha=0.7)
    for turn in where_auto_recomputes:
        axis.axvline(turn, color="C1", linestyle="-", alpha=0.2)

axs["epsx"].plot(turns, 1e10 * pd.Series(regular.epsilon_x).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=2, label=f"Fixed ({ibs_step}) turns")
axs["epsy"].plot(turns, 1e13 * pd.Series(regular.epsilon_y).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=2, label=f"Fixed ({ibs_step}) turns")
axs["sigd"].plot(turns, 1e3 * pd.Series(regular.sigma_delta).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=2, label=f"Fixed ({ibs_step}) turns")
axs["bl"].plot(turns, 1e3 * pd.Series(regular.bunch_length).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=2, label=f"Fixed ({ibs_step}) turns")

axs["epsx"].plot(turns, 1e10 * pd.Series(auto.epsilon_x).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=1.9, label=f"Auto ({AUTO_PERCENT:.0f}% change)")
axs["epsy"].plot(turns, 1e13 * pd.Series(auto.epsilon_y).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=1.9, label=f"Auto ({AUTO_PERCENT:.0f}% change)")
axs["sigd"].plot(turns, 1e3 * pd.Series(auto.sigma_delta).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=1.9, label=f"Auto ({AUTO_PERCENT:.0f}% change)")
axs["bl"].plot(turns, 1e3 * pd.Series(auto.bunch_length).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=1.9, label=f"Auto ({AUTO_PERCENT:.0f}% change)")

# Axes parameters
axs["epsx"].set_ylabel(r"$\varepsilon_x$ [$10^{-10}$m]")
axs["epsy"].set_ylabel(r"$\varepsilon_y$ [$10^{-13}$m]")
axs["sigd"].set_ylabel(r"$\sigma_{\delta}$ [$10^{-3}$]")
axs["bl"].set_ylabel(r"Bunch length [mm]")

for axis in (axs["epsy"], axs["bl"]):
    axis.yaxis.set_label_position("right")
    axis.yaxis.tick_right()

for axis in (axs["sigd"], axs["bl"]):
    axis.set_xlabel("Turn Number")

for axis in axs.values():
    axis.yaxis.set_major_locator(plt.MaxNLocator(3))

fig.align_ylabels((axs["epsx"], axs["sigd"]))
fig.align_ylabels((axs["epsy"], axs["bl"]))
fig.suptitle(f"Evolution of Emittances in CLIC DR (Kinetic Kicks)\nAverage of Last {AVG_TURNS} Turns")

plt.legend(title="Recompute Coefficients")
plt.tight_layout()
plt.show()


###############################################################################
# We can have a look at the relative change from turn to turn:

fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(13, 7))

# We will add vertical lines at the times where recomputing of the kick 
# coefficients happened (do this first so they show up in the background)
for axis in axs.values():
    for turn in where_fixed_recomputes:
        axis.axvline(turn, color="gray", linestyle="--", lw=1, alpha=0.7)
    for turn in where_auto_recomputes:
        axis.axvline(turn, color="C1", linestyle="-", alpha=0.3)

axs["epsx"].plot(turns, 1e2 * pd.Series(regular.epsilon_x).pct_change(), lw=1, label=f"Fixed ({ibs_step}) turns")
axs["epsy"].plot(turns, 1e2 * pd.Series(regular.epsilon_y).pct_change(), lw=1, label=f"Fixed ({ibs_step}) turns")
axs["sigd"].plot(turns, 1e2 * pd.Series(regular.sigma_delta).pct_change(), lw=1, label=f"Fixed ({ibs_step}) turns")
axs["bl"].plot(turns, 1e2 * pd.Series(regular.bunch_length).pct_change(), lw=1, label=f"Fixed ({ibs_step}) turns")

axs["epsx"].plot(turns, 1e2 * pd.Series(auto.epsilon_x).pct_change(), lw=1, label=f"Auto ({AUTO_PERCENT:.0f}% change)")
axs["epsy"].plot(turns, 1e2 * pd.Series(auto.epsilon_y).pct_change(), lw=1, label=f"Auto ({AUTO_PERCENT:.0f}% change)")
axs["sigd"].plot(turns, 1e2 * pd.Series(auto.sigma_delta).pct_change(), lw=1, label=f"Auto ({AUTO_PERCENT:.0f}% change)")
axs["bl"].plot(turns, 1e2 * pd.Series(auto.bunch_length).pct_change(), lw=1, label=f"Auto ({AUTO_PERCENT:.0f}% change)")

# Axes parameters
axs["epsx"].set_ylabel(r"$\varepsilon_x$")
axs["epsy"].set_ylabel(r"$\varepsilon_y$")
axs["sigd"].set_ylabel(r"$\sigma_{\delta}$")
axs["bl"].set_ylabel(r"Bunch length")

for axis in (axs["epsy"], axs["bl"]):
    axis.yaxis.set_label_position("right")
    axis.yaxis.tick_right()

for axis in (axs["sigd"], axs["bl"]):
    axis.set_xlabel("Duration [h]")

for axis in axs.values():
    axis.axhline(AUTO_PERCENT, color="black", linestyle="--", alpha=0.5, label="Recompute Threshold")
    axis.yaxis.set_major_locator(plt.MaxNLocator(3))
    axis.set_yscale("log")

fig.align_ylabels((axs["epsx"], axs["sigd"]))
fig.align_ylabels((axs["epsy"], axs["bl"]))
fig.suptitle("Percent change from previous turn")

plt.legend(title="Recompute Coefficients")
plt.tight_layout()
plt.show()


###############################################################################
# We can see that the auto-recompute feature - while leading to additional compute
# from a higher number of kick coefficients computation - helped not overestimate
# the IBS effects in the horizontal plane, and not underestimate them in the 
# vertical plane, in addition to making the tracking loop simpler to write.


#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `~xibs.analytical`: `~.xibs.analytical.NagaitsevIBS`, `~.xibs.analytical.NagaitsevIBS.growth_rates`, `~.xibs.analytical.NagaitsevIBS.emittance_evolution`
#    - `~xibs.inputs`: `~xibs.inputs.BeamParameters`, `~xibs.inputs.OpticsParameters`
#    - `~xibs.kicks`: `~.xibs.kicks.DiffusionCoefficients`, `~.xibs.kicks.FrictionCoefficients`, `~.xibs.kicks.IBSKickCoefficients`, `~.xibs.kicks.KineticKickIBS`, `~.xibs.kicks.KineticKickIBS.apply_ibs_kick`, `~.xibs.kicks.KineticKickIBS.compute_kick_coefficients`

###############################################################################
