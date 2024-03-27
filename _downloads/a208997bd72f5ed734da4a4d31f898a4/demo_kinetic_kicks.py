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
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt

from xibs.analytical import NagaitsevIBS
from xibs.formulary import _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta
from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import DiffusionCoefficients, KineticKickIBS

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
# of particles in the bunch, and its value influences the IBS kick coefficients
# calculation while `n_part` is the number of generated particles for
# tracking, which is much lower.

line_file = "lines/chrom-corr_DR.newlattice_2GHz.json"
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

particles = xp.generate_matched_gaussian_bunch(
    num_particles=n_part,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    line=line,
)

###############################################################################
# Generating the IBS Kick Class
# -----------------------------
# Just as all user-facing classes in ``xibs``, the `KineticKickIBS` :cite:`NuclInstr:Zenkevich:Kinetic_IBS`
# class is instantiated by providing a `BeamParameters` and an `OpticsParameters` objects:

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

###############################################################################
# Computing and Applying IBS Kicks
# --------------------------------
# Since calculating the IBS kicks requires computing the diffusion and friction rates,
# which can be computationally expensive and not necessary every turn, this functionality
# is distinct from the application of the kicks to the particles. The so-called "kick
# coefficients" are computed directly from the particle distribution by calling the
# dedicated method. They correspond to the diffusion minus friction terms, and are
# returned as an `IBSKickCoefficients` object, which is stored internally in the **IBS**
# object and will be updated internally each time they are computed.

kick_coefficients = IBS.compute_kick_coefficients(particles)
print(kick_coefficients)
print(IBS.kick_coefficients)

###############################################################################
# For the kinetic formalism, the diffusion and friction coefficients are also
# stored internally (and updated each time the method above is called). They
# can be accessed as the `diffusion_coefficients` and `friction_coefficients`
# attributes of the `KineticKickIBS` object.
# 
# .. hint::
#     Note that in this formalism the order of magnitude of diffusion and friction
#     coefficients, as well as the resulting `IBSKickCoefficients`` is very different
#     than the ones from the simple kicks formalism.

print(IBS.diffusion_coefficients)
print(IBS.friction_coefficients)

###############################################################################
# We can also manually set arbitrary values through these attributes. We will do
# so here with unrealistically high diffusion values compared to the ones above,
# in order to demonstrate the effect of the kick. The kick is applied to the particles
# with the `apply_ibs_kick` method, which computes and applies momentum kicks on each
# plane, for both diffusion and friction effects, weigthed by the line density of the
# bunch. Friction terms lead to a reduction of the beam sizes, while diffusion terms
# lead to an increase.

# Let's do strong diffusion and small friction to showcase the effect
IBS.diffusion_coefficients = DiffusionCoefficients(5e5, 1e6, 5e5)
particles2 = particles.copy()  # let's apply on a copy of the particles
IBS.apply_ibs_kick(particles2)

###############################################################################
# We can have a look at the effect on the particles (see the `xsuite user guide
# <https://xsuite.readthedocs.io/en/latest/particlesmanip.html>`_ for more)

fig, (axx, axy, axz) = plt.subplots(1, 3, figsize=(15, 5))

axx.plot(1e4 * particles2.x, 1e4 * particles2.px, ".", label="After Kick")
axx.plot(1e4 * particles.x, 1e4 * particles.px, ".", label="Before Kick")

axy.plot(1e6 * particles2.y, 1e6 * particles2.py, ".", label="After Kick")
axy.plot(1e6 * particles.y, 1e6 * particles.py, ".", label="Before Kick")

axz.plot(1e3 * particles2.zeta, 1e3 * particles2.delta, ".", label="After Kick")
axz.plot(1e3 * particles.zeta, 1e3 * particles.delta, ".", label="Before Kick")

axx.set_xlabel(r"$x$ [$10^{-4}$m]")
axx.set_ylabel(r"$p_x$ [$10^{-4}$]")
axy.set_xlabel(r"$y$ [$10^{-6}$m]")
axy.set_ylabel(r"$p_y$ [$10^{-6}$]")
axz.set_xlabel(r"$z$ [$10^{-3}$]")
axz.set_ylabel(r"$\delta$ [$10^{-3}$]")

for axis in (axx, axy, axz):
    axis.yaxis.set_major_locator(plt.MaxNLocator(3))
    axis.legend()

plt.tight_layout()
plt.show()

###############################################################################
# Applying IBS Kicks in Tracking
# ------------------------------
# Let's now include this computation and application of IBS kicks in a tracking
# simulation. We will first re-initialize the particle distribution with less
# individual particles to speed up the tracking, and re-initialize IBS classes.
# We will also track the analytical evolution of relevant quantities - as done
# in the :ref:`Nagaitsev example <demo-analytical-nagaitsev>` - for comparison.

IBS = KineticKickIBS(beamparams, opticsparams)
NIBS = NagaitsevIBS(beamparams, opticsparams)

# Re-create particles with less elements as tracking takes a while
n_part = int(1.5e3)  # 1500 particles should be enough for this example
particles = xp.generate_matched_gaussian_bunch(
    num_particles=n_part,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    line=line,
)

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


# Initialize the dataclasses & store the initial values
kicked_tbt = Records.init_zeroes(nturns)
analytical_tbt = Records.init_zeroes(nturns)

kicked_tbt.update_at_turn(0, particles, twiss)
analytical_tbt.update_at_turn(0, particles, twiss)

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
        print(f"Turn {turn:d}: re-computing diffusion and friction terms")
        # Compute kick coefficients from the particle distribution at this moment
        IBS.compute_kick_coefficients(particles)
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
    line.track(particles, num_turns=1)

    # ----- Update records for tracked particles ----- #
    kicked_tbt.update_at_turn(turn, particles, twiss)

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

###############################################################################
# Feel free to run this simulation for more turns, with a different frequency of
# the IBS kick coefficients & growth rates re-computation, or with more particles.
# After this loop is done running, we can plot the evolutions across turns:

turns = np.arange(nturns, dtype=int)  # array of turns
fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(15, 7))

# Plot from tracked & kicked particles
axs["epsx"].plot(turns, kicked_tbt.epsilon_x * 1e10, lw=2, label="Kinetic Kicks")
axs["epsy"].plot(turns, kicked_tbt.epsilon_y * 1e13, lw=2, label="Kinetic Kicks")
axs["sigd"].plot(turns, kicked_tbt.sigma_delta * 1e3, lw=2, label="Kinetic Kicks")
axs["bl"].plot(turns, kicked_tbt.bunch_length * 1e3, lw=2, label="Kinetic Kicks")

# Plot from analytical values
axs["epsx"].plot(turns, analytical_tbt.epsilon_x * 1e10, lw=2.5, label="Analytical")
axs["epsy"].plot(turns, analytical_tbt.epsilon_y * 1e13, lw=2.5, label="Analytical")
axs["sigd"].plot(turns, analytical_tbt.sigma_delta * 1e3, lw=2.5, label="Analytical")
axs["bl"].plot(turns, analytical_tbt.bunch_length * 1e3, lw=2.5, label="Analytical")

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
#    - `~xibs.analytical`: `~.xibs.analytical.NagaitsevIBS`, `~.xibs.analytical.NagaitsevIBS.growth_rates`, `~.xibs.analytical.NagaitsevIBS.emittance_evolution`
#    - `~xibs.inputs`: `~xibs.inputs.BeamParameters`, `~xibs.inputs.OpticsParameters`
#    - `~xibs.kicks`: `~.xibs.kicks.DiffusionCoefficients`, `~.xibs.kicks.FrictionCoefficients`, `~.xibs.kicks.IBSKickCoefficients`, `~.xibs.kicks.KineticKickIBS`, `~.xibs.kicks.KineticKickIBS.apply_ibs_kick`, `~.xibs.kicks.KineticKickIBS.compute_kick_coefficients`

###############################################################################
