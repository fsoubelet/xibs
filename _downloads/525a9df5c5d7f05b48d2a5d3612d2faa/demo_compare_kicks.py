"""

.. _demo-compare-kicks:

==================================
Comparison of IBS Kicks Formalisms
==================================

This example shows a comparison of the results obtained with `~xibs.kicks.KineticKickIBS`
and `~xibs.kicks.SimpleKickIBS` when applying IBS kicks to tracked particles.

We will do the comparison by using the case of the CERN SPS, with lead ions at top
energy, and with some tweaked beam parameters to stimulate the IBS effects.

We will demonstrate using an `xtrack.Line` of the ``SPS`` ring, for ions at top energy (450 GeV).
"""
# sphinx_gallery_thumbnail_number = 1
import logging
import sys
import warnings

from dataclasses import dataclass
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt

from xibs.inputs import BeamParameters, OpticsParameters
from xibs.formulary import _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta
from xibs.kicks import KineticKickIBS, SimpleKickIBS

warnings.simplefilter("ignore")  # for this tutorial's clarity
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
# of particles in the bunch, and its value influences the analytical IBS growth
# rates calculation while `n_part` is the number of generated particles for
# tracking, which is much lower.

line_file = "lines/sps_top_ions.json"
bunch_intensity = int(5e11)
n_part = int(1e3)
sigma_z = 5e-2
nemitt_x = 1.0e-6
nemitt_y = 0.25e-6

###############################################################################
# Setting up line and particles
# -----------------------------
# Let's now load the line from file, activate the accelerating cavities and
# create a context for multithreading with OpenMP, since tracking particles
# is going to take some time:

line = xt.Line.from_json(line_file)  # includes reference particle
context = xo.ContextCpu(omp_num_threads="auto")


# ----- Power accelerating cavities ----- #
rf_voltage = 1.7e6  # 1.7MV
harmonic_number = 4653
cavity = "actcse.31632"
line[cavity].lag = 180  # 180 above transition
line[cavity].voltage = rf_voltage
line[cavity].frequency = OpticsParameters.from_line(line).revolution_frequency * harmonic_number

line.build_tracker(context)
line.optimize_for_tracking()
twiss = line.twiss(method="4d")

# We create a particle distribution which we clone, one for each kick formalism
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
# Generating the IBS Kick Classes
# -------------------------------
# Just as all user-facing classes in ``xibs``, the `SimpleKickIBS` :cite:`PRAB:Bruce:Simple_IBS_Kicks`
# class is instantiated by providing a `BeamParameters` and an `OpticsParameters` objects:

beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
opticsparams = OpticsParameters.from_line(line)
KIBS = KineticKickIBS(beamparams, opticsparams)
SIBS = SimpleKickIBS(beamparams, opticsparams)

###############################################################################
# Comparing Beam Parameters Evolution While Applying IBS Kicks in Tracking
# ------------------------------------------------------------------------
# We will now apply kicks with both formalism during tracking simulations.
# Let's do so for 750 turns, and re-compute kick coefficients every 50 turns.

# Define some parameters for the tracking
nturns = 750  # number of turns to loop for
ibs_step = 50  # frequency at which to re-compute coefficients in [turns]

###############################################################################
# We will also set up a dataclass conveniently to store the results:

@dataclass
class Records:
    epsilon_x: np.ndarray
    epsilon_y: np.ndarray
    sigma_delta: np.ndarray
    bunch_length: np.ndarray

    def update_at_turn(self, turn: int, parts: xp.Particles, twiss: xt.TwissTable):
        """Automatically update the records at given turn from the xtrack.Particles."""
        self.epsilon_x[turn] = _geom_epsx(parts, twiss.betx[0], twiss.dx[0])
        self.epsilon_y[turn] = _geom_epsy(parts, twiss.bety[0], twiss.dy[0])
        self.sigma_delta[turn] = _sigma_delta(parts)
        self.bunch_length[turn] = _bunch_length(parts)

    @classmethod
    def init_zeroes(cls, n_turns: int) -> Self:  # noqa: F821
        """Initialize the dataclass with arrays of zeroes."""
        return cls(
            epsilon_x=np.zeros(n_turns, dtype=float),
            epsilon_y=np.zeros(n_turns, dtype=float),
            sigma_delta=np.zeros(n_turns, dtype=float),
            bunch_length=np.zeros(n_turns, dtype=float),
        )


# Initialize the dataclasses & store the initial values
kinetic_tbt = Records.init_zeroes(nturns)
simple_tbt = Records.init_zeroes(nturns)

kinetic_tbt.update_at_turn(0, particles, twiss)
simple_tbt.update_at_turn(0, particles2, twiss)

# Let's hide anything below WARNING level for readability
logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stdout,
    format="[%(asctime)s] [%(levelname)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

###############################################################################
# Now, since ``xibs`` is not fully integrated into Xsuite, we will have to manually
# apply the IBS kick at each turn of tracking, and also manually trigger the turn
# of tracking. Just like in the analytical examples, we do so in a loop over the turns:

# ----- We loop here now ----- # 

for turn in range(1, nturns):
    # ----- Potentially re-compute the IBS growth rates and kick coefficients ----- #
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"Turn {turn:d}: re-computing IBS kick terms")
        # Compute kick coefficients from the particle distributions at this moment
        KIBS.compute_kick_coefficients(particles)
        SIBS.compute_kick_coefficients(particles2)
    else:
        print(f"Turn {turn:d}")

    # ----- Manually Apply IBS Kick and Track Turn ----- #
    KIBS.apply_ibs_kick(particles)
    SIBS.apply_ibs_kick(particles2)
    line.track(particles, num_turns=1)
    line.track(particles2, num_turns=1)

    # ----- Update records for tracked particles ----- #
    kinetic_tbt.update_at_turn(turn, particles, twiss)
    simple_tbt.update_at_turn(turn, particles2, twiss)

###############################################################################
# Feel free to run this simulation for more turns, with a different frequency
# of the IBS kick coefficients re-computation, or with more particles. After
# this loop is done running, we can plot the evolutions across turns:

turns = np.arange(nturns, dtype=int)  # array of turns
fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(15, 7))

# Plot from kinetic kicks
axs["epsx"].plot(turns, kinetic_tbt.epsilon_x * 1e10, lw=2, label="Kinetic")
axs["epsy"].plot(turns, kinetic_tbt.epsilon_y * 1e13, lw=2, label="Kinetic")
axs["sigd"].plot(turns, kinetic_tbt.sigma_delta * 1e3, lw=2, label="Kinetic")
axs["bl"].plot(turns, kinetic_tbt.bunch_length * 1e3, lw=2, label="Kinetic")

# Plot from simple kicks
axs["epsx"].plot(turns, simple_tbt.epsilon_x * 1e10, lw=2, label="Simple")
axs["epsy"].plot(turns, simple_tbt.epsilon_y * 1e13, lw=2, label="Simple")
axs["sigd"].plot(turns, simple_tbt.sigma_delta * 1e3, lw=2, label="Simple")
axs["bl"].plot(turns, simple_tbt.bunch_length * 1e3, lw=2, label="Simple")

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
    axis.legend(loc=9, ncols=4)

fig.align_ylabels((axs["epsx"], axs["sigd"]))
fig.align_ylabels((axs["epsy"], axs["bl"]))

plt.tight_layout()
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `~xibs.inputs`: `~xibs.inputs.BeamParameters`, `~xibs.inputs.OpticsParameters`
#    - `~xibs.kicks`: `~.xibs.kicks.KineticKickIBS`, `~.xibs.kicks.SimpleKickIBS`, `~.xibs.kicks.KineticKickIBS.apply_ibs_kick`, `~.xibs.kicks.KineticKickIBS.compute_kick_coefficients`

###############################################################################
