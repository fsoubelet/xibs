"""

.. _demo-analytical-bjorken-mtingwa:

===========================================================================
Bjorken-Mtingwa Formalism - Analytical Growth Rates and Emittance Evolution
===========================================================================

This example shows how to use the `~.xibs.analytical.BjorkenMtingwaIBS` class
to calculate IBS growth rates and emittances evolutions analytically.

We will demonstrate using the case of the CERN SPS, with protons at top
energy, aka 450 GeV.
"""
# sphinx_gallery_thumbnail_number = 2
import logging
import warnings

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import xpart as xp
import xtrack as xt

from xibs.analytical import BjorkenMtingwaIBS
from xibs.formulary import _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta
from xibs.inputs import BeamParameters, OpticsParameters

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
# Let's start by defining the line and particle information, as well as some
# parameters for later use:

line_file = "lines/sps_top_protons.json"
harmonic_number = 4653
cavity_name = "actcse.31632"  # from xsuite documentation examples
cavity_lag = 180  # from xsuite documentation examples
rf_frequency = 200e6  # from xsuite documentation examples
rf_voltage = 4  # in MV,  from xsuite documentation examples
energy_loss = 0  # let's pretend ;)
bunch_intensity = 1.2e11
sigma_z = 22.5e-2  # from xsuite documentation examples
nemitt_x = 2e-6  # from xsuite documentation examples
nemitt_y = 2.5e-6  # from xsuite documentation examples
n_part = int(5e3)  # for this example let's not use too many particles

###############################################################################
# Setting up line and particles
# -----------------------------
# Let's start by loading the `xtrack.Line`, activating RF cavities and
# generating a matched Gaussian bunch:

line = xt.Line.from_json(line_file)
line.build_tracker()  # The line.particle_ref is included in the file

# ----- Power accelerating cavity - from xsuite documentation examples ----- #
line[cavity_name].voltage = rf_voltage * 1e6  # to be given in [V] here
line[cavity_name].lag = cavity_lag
line[cavity_name].frequency = rf_frequency
twiss = line.twiss()

particles = xp.generate_matched_gaussian_bunch(
    num_particles=n_part,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    particle_ref=line.particle_ref,
    line=line,
)

###############################################################################
# We can have a look at the generated particles (see the `xsuite user guide
# <https://xsuite.readthedocs.io/en/latest/particlesmanip.html>`_ for more)

fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(9, 11))

axx.plot(1e6 * particles.x, 1e5 * particles.px, ".", ms=3)
axy.plot(1e6 * particles.y, 1e6 * particles.py, ".", ms=3)
axz.plot(1e3 * particles.zeta, 1e3 * particles.delta, ".", ms=3)

axx.set_xlabel(r"$x$ [$\mu$m]")
axx.set_ylabel(r"$p_x$ [$10^{-5}$]")
axy.set_xlabel(r"$y$ [$\mu$m]")
axy.set_ylabel(r"$p_y$ [$10^{-3}$]")
axz.set_xlabel(r"$z$ [$10^{-3}$]")
axz.set_ylabel(r"$\delta$ [$10^{-3}$]")
fig.align_ylabels([axx, axy, axz])
plt.tight_layout()
plt.show()

###############################################################################
# We can compute initial (geometrical) emittances as well as the bunch length
# from the `xtrack.Particles` object:

geom_epsx = _geom_epsx(particles, twiss.betx[0], twiss.dx[0])
geom_epsy = _geom_epsy(particles, twiss.bety[0], twiss.dy[0])
bunch_l = _bunch_length(particles)
sig_delta = _sigma_delta(particles)

###############################################################################
# Computing IBS Growth Rates
# --------------------------
# Let us instantiate the `~.xibs.analytical.BjorkenMtingwaIBS` class. It is fairly
# simple and is done from both the beam parameters of the particles and the
# optics parameters of the line. For each of these a specific `dataclass`
# is provided in the `~.xibs.inputs` module.

beam_params = BeamParameters(particles)
optics = OpticsParameters(twiss)
IBS = BjorkenMtingwaIBS(beam_params, optics)

###############################################################################
# Just like for NagaitsevIBS, we can ask for the **IBS** growth rates to be computed
# with a dedicated function. Notice the class will complain that our provided Twiss
# was not computed at the center of elements, but still perform the calculation. At
# the moment xtrack does not allow Twissing at the center of elements, but we could
# get this from MAD-X. These are returned by the function but also stored and updated
# internally each time they are computed.

growth_rates = IBS.growth_rates(geom_epsx, geom_epsy, sig_delta, bunch_l)
print(growth_rates)
print(IBS.ibs_growth_rates)

###############################################################################
# Computing New Emittances from Growth Rates
# ------------------------------------------
# From these one can compute the emittances at the next time step. For the emittances
# at the next turn, once should use :math:`1 / f_{rev}` as the time step, which
# is the default value used if none is provided.

new_geom_epsx, new_geom_epsy, new_sig_delta, new_bunch_length = IBS.emittance_evolution(
    geom_epsx, geom_epsy, sig_delta, bunch_l
)
print(f"Next time step geometrical epsilon_x = {new_geom_epsx} m")
print(f"Next time step geometrical epsilon_y = {new_geom_epsy} m")
print(f"Next time step sigma_delta = {new_sig_delta}")
print(f"Next time step bunch length = {new_bunch_length} m")

###############################################################################
# Analytical Evolution for a Time Period
# --------------------------------------
# One can then analytically look at the evolution through time by looping
# over this calculation. Let's do this for 5 hours, recomputing the growth rates
# every 10 minutes. The more frequent this update the more physically accurate
# the results will be, but the longer the simulation as this computation is the most
# compute-intensive process.

nsecs = 5 * 3_600  # that's 5h
ibs_step = 10 * 60  # re-compute rates every 10min
seconds = np.linspace(0, nsecs, nsecs).astype(int)


# Set up a dataclass to store the results
@dataclass
class Records:
    """Dataclass to store (and update) important values through tracking."""

    epsilon_x: np.ndarray
    epsilon_y: np.ndarray
    sig_delta: np.ndarray
    bunch_length: np.ndarray

    @classmethod
    def init_zeroes(cls, n_turns: int):
        return cls(
            epsilon_x=np.zeros(n_turns, dtype=float),
            epsilon_y=np.zeros(n_turns, dtype=float),
            sig_delta=np.zeros(n_turns, dtype=float),
            bunch_length=np.zeros(n_turns, dtype=float),
        )

    def update_at_turn(self, turn: int, epsx: float, epsy: float, sigd: float, bl: float):
        """Works for turns / seconds, just needs the correct index to store in."""
        self.epsilon_x[turn] = epsx
        self.epsilon_y[turn] = epsy
        self.sig_delta[turn] = sigd
        self.bunch_length[turn] = bl

# Initialize the dataclass & store the initial values
turn_by_turn = Records.init_zeroes(nsecs)
turn_by_turn.update_at_turn(0, geom_epsx, geom_epsy, sig_delta, bunch_l)


# ----- We loop here now ----- # 

for sec in range(1, nsecs):
    # ----- Potentially re-compute the IBS growth rates ----- #
    if (sec % ibs_step == 0) or (sec == 1):
        print(f"At {sec}s: re-computing growth rates")
        # We compute from values at the previous turn
        IBS.growth_rates(
            turn_by_turn.epsilon_x[sec - 1],
            turn_by_turn.epsilon_y[sec - 1],
            turn_by_turn.sig_delta[sec - 1],
            turn_by_turn.bunch_length[sec - 1],
        )

    # ----- Compute the new emittances ----- #
    new_emit_x, new_emit_y, new_sig_delta, new_bunch_length = IBS.emittance_evolution(
        epsx=turn_by_turn.epsilon_x[sec - 1],
        epsy=turn_by_turn.epsilon_y[sec - 1],
        sigma_delta=turn_by_turn.sig_delta[sec - 1],
        bunch_length=turn_by_turn.bunch_length[sec - 1],
        dt=1.0,  # get at next second
    )

    # ----- Update the records with the new values ----- #
    turn_by_turn.update_at_turn(sec, new_emit_x, new_emit_y, new_sig_delta, new_bunch_length)


###############################################################################
# Feel free to run this simulation for longer, or with a different frequency
# of the IBS growth rates re-computation. After this is done running, we can plot
# the evolutions across the turns:

fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(13, 7))

axs["epsx"].plot(seconds / 3600, 1e9 * turn_by_turn.epsilon_x, lw=2)
axs["epsy"].plot(seconds / 3600, 1e9 * turn_by_turn.epsilon_y, lw=2)
axs["sigd"].plot(seconds / 3600, 1e4 * turn_by_turn.sig_delta, lw=2)
axs["bl"].plot(seconds / 3600, 1e2 * turn_by_turn.bunch_length, lw=2)

# Axes parameters
axs["epsx"].set_ylabel(r"$\varepsilon_x$ [$10^{-9}$m]")
axs["epsy"].set_ylabel(r"$\varepsilon_y$ [$10^{-9}$m]")
axs["sigd"].set_ylabel(r"$\sigma_{\delta}$ [$10^{-4}$]")
axs["bl"].set_ylabel(r"Bunch length [cm]")

for axis in (axs["epsy"], axs["bl"]):
    axis.yaxis.set_label_position("right")
    axis.yaxis.tick_right()

for axis in (axs["sigd"], axs["bl"]):
    axis.set_xlabel("Duration [h]")

for axis in axs.values():
    axis.yaxis.set_major_locator(plt.MaxNLocator(3))

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
#    - `~xibs.analytical`: `~.xibs.analytical.BjorkenMtingwaIBS`, `~.xibs.analytical.BjorkenMtingwaIBS.growth_rates`, `~.xibs.analytical.BjorkenMtingwaIBS.emittance_evolution`
#    - `~xibs.inputs`: `~xibs.inputs.BeamParameters`, `~xibs.inputs.OpticsParameters`

###############################################################################
