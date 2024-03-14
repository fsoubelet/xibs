"""

.. _demo-kinetic-kicks:

=========================================================================
Kinetic Kicks Formalism - IBS Kicks Based on Diffusion and Friction Terms
=========================================================================

This example shows how to use the `~.xibs.kicks.KineticKickIBS` class
to apply IBS kicks to tracked particles based on the kinetic theory of gases.
It implements IBS kicks based on the formalism described in :cite:`NuclInstr:Zenkevich:Kinetic_IBS`.

.. warning::
    Please note that this kick formalism is currently implemented with terms from Nagaitsev's
    calculations, which do not take into consideration vertical dispersion. Details are provided
    in the :ref:`class's documentation <xibs-kicks>`.

We will demonstrate using an `xtrack.Line` of the ``CLIC`` damping ring,
for a positron beam.
"""
# sphinx_gallery_thumbnail_number = 2
import logging
import sys
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt

from xibs.analytical import NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS
from xibs._old_michalis import MichalisIBS

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
# We will define here some helper functions we will regularly use later on to
# compute the emittances, bunch lengths and momentum spread from and `xpart.Particles`
# object representing our particle distribution. In each, let's make sure to filter
# for only the active particles (``state > 0``).


def _bunch_length(parts: xp.Particles) -> float:
    return np.std(parts.zeta[parts.state > 0])


def _sigma_delta(parts: xp.Particles) -> float:
    return np.std(parts.delta[parts.state > 0])


def _geom_epsx(parts: xp.Particles, twiss: xt.TwissTable) -> float:
    """
    We index dx and betx at 0 which corresponds to the beginning / end of
    the line, since this is where / when we will be applying the kicks.
    """
    sigma_x = np.std(parts.x[parts.state > 0])
    sig_delta = _sigma_delta(parts)
    return (sigma_x**2 - (twiss["dx"][0] * sig_delta) ** 2) / twiss["betx"][0]


def _geom_epsy(parts: xp.Particles, twiss: xt.TwissTable) -> float:
    """
    We index dy and bety at 0 which corresponds to the beginning / end of
    the line, since this is where / when we will be applying the kicks.
    """
    sigma_y = np.std(parts.y[parts.state > 0])
    sig_delta = _sigma_delta(parts)
    return (sigma_y**2 - (twiss["dy"][0] * sig_delta) ** 2) / twiss["bety"][0]


###############################################################################
# Let's start by defining the line and particle information, as well as some
# parameters for later use. Note that `bunch_intensity` is the actual number
# of particles in the bunch, and its value influences the analytical IBS growth
# rates calculation while `n_part` is the number of generated particles for
# tracking, which is much lower.

line_file = Path(__file__).parent / "lines/chrom-corr_DR.newlattice_2GHz.json"
bunch_intensity = int(4.5e9)
n_part = int(1.5e4)  # 15k particles initially to have a look at the kick effect
sigma_z = 1.58e-3
nemitt_x = 5.66e-7
nemitt_y = 3.7e-9

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

# Let's make sure we get logging output to demonstrate
logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stdout,
    format="[%(asctime)s] [%(levelname)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
)

beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
opticsparams = OpticsParameters.from_line(line)


IBS = KineticKickIBS(beamparams, opticsparams)
NIBS = NagaitsevIBS(beamparams, opticsparams)

# Re-create particles with less elements as tracking takes a while
n_part = int(2.5e3)  # 2500 particles should be enough for this example
particles = xp.generate_matched_gaussian_bunch(
    num_particles=n_part,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    line=line,
)
particles2 = particles.copy()

MIBS = MichalisIBS()
MIBS.set_beam_parameters(particles)
MIBS.set_optic_functions(twiss)

# Define some parameters for the tracking
nturns = 1500  # number of turns to loop for
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
        """Automatically update the records at given turn from the xpart.Particles."""
        self.epsilon_x[turn] = _geom_epsx(parts, twiss)
        self.epsilon_y[turn] = _geom_epsy(parts, twiss)
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


# Initialize the dataclasses
kicked_tbt = Records.init_zeroes(nturns)
old_tbt = Records.init_zeroes(nturns)
analytical_tbt = Records.init_zeroes(nturns)

# Store the initial values
kicked_tbt.update_at_turn(0, particles, twiss)
old_tbt.update_at_turn(0, particles, twiss)
analytical_tbt.update_at_turn(0, particles, twiss)

# We loop here now
for turn in range(1, nturns):
    # ----- Potentially re-compute the IBS growth rates and kick coefficients ----- #
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"Turn {turn:d}: re-computing diffusion and friction terms")
        # Compute kick coefficients from the particle distribution at this moment
        IBS.compute_kick_coefficients(particles)
        MIBS.calculate_kinetic_coefficients(particles2)
        # Compute analytical values from those at the previous turn
        NIBS.growth_rates(
            analytical_tbt.epsilon_x[turn - 1],
            analytical_tbt.epsilon_y[turn - 1],
            analytical_tbt.sigma_delta[turn - 1],
            analytical_tbt.bunch_length[turn - 1],
        )
    else:
        print(f"Turn {turn:d}")

    # ----- Manually Apply IBS Kick and Track Turn ----- #
    IBS.apply_ibs_kick(particles)
    MIBS.apply_kinetic_kick(particles2)
    line.track(particles, num_turns=1)
    line.track(particles2, num_turns=1)

    # ----- Update records for tracked particles ----- #
    kicked_tbt.update_at_turn(turn, particles, twiss)
    old_tbt.update_at_turn(turn, particles2, twiss)

    # ----- Compute analytical Emittances from previous turn values & update records----- #
    ana_emit_x, ana_emit_y, ana_sig_delta, ana_bunch_length = NIBS.emittance_evolution(
        analytical_tbt.epsilon_x[turn - 1],
        analytical_tbt.epsilon_y[turn - 1],
        analytical_tbt.sigma_delta[turn - 1],
        analytical_tbt.bunch_length[turn - 1],
    )
    analytical_tbt.epsilon_x[turn] = ana_emit_x
    analytical_tbt.epsilon_y[turn] = ana_emit_y
    analytical_tbt.sigma_delta[turn] = ana_sig_delta
    analytical_tbt.bunch_length[turn] = ana_bunch_length


turns = np.arange(nturns, dtype=int)  # array of turns
fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(15, 7.5))

# Plot from tracked & kicked particles
l1, = axs["epsx"].plot(turns, kicked_tbt.epsilon_x * 1e10, lw=2, label="Xibs")
axs["epsy"].plot(turns, kicked_tbt.epsilon_y * 1e13, lw=2, label="Xibs")
axs["sigd"].plot(turns, kicked_tbt.sigma_delta * 1e3, lw=2, label="Xibs")
axs["bl"].plot(turns, kicked_tbt.bunch_length * 1e3, lw=2, label="Xibs")

# Plot from michalis results
l2, = axs["epsx"].plot(turns, old_tbt.epsilon_x * 1e10, lw=2, label="Michalis")
axs["epsy"].plot(turns, old_tbt.epsilon_y * 1e13, lw=2, label="Michalis")
axs["sigd"].plot(turns, old_tbt.sigma_delta * 1e3, lw=2, label="Michalis")
axs["bl"].plot(turns, old_tbt.bunch_length * 1e3, lw=2, label="Michalis")

# Plot from analytical values
l3, = axs["epsx"].plot(turns, analytical_tbt.epsilon_x * 1e10, lw=2.5, label="Nagaitsev")
axs["epsy"].plot(turns, analytical_tbt.epsilon_y * 1e13, lw=2.5, label="Nagaitsev")
axs["sigd"].plot(turns, analytical_tbt.sigma_delta * 1e3, lw=2.5, label="Nagaitsev")
axs["bl"].plot(turns, analytical_tbt.bunch_length * 1e3, lw=2.5, label="Nagaitsev")

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
    # axis.legend(loc=9, ncols=4)

fig.align_ylabels((axs["epsx"], axs["sigd"]))
fig.align_ylabels((axs["epsy"], axs["bl"]))

fig.suptitle("CLIC DR - Kinetic Kicks")
fig.legend(handles=(l1, l2, l3), loc="lower center", bbox_to_anchor=(0.5, 0.873), ncol=3, fontsize=14)

plt.tight_layout()
plt.show()
