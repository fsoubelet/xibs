"""

.. _demo-analytical:

===============================================
Analytical Growth Rates and Emittance Evolution
===============================================

This example shows how to use the `~.xibs.analytical.NagaitsevIBS` class
to calculate IBS growth rates and emittances evolutions analytically.

We will demonstrate using an `xtrack.Line` of the ``CLIC`` damping ring,
for a positron beam.
"""
# sphinx_gallery_thumbnail_number = 2
import json
import warnings

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import xpart as xp
import xtrack as xt

from xibs.analytical import NagaitsevIBS
from xibs.formulary import bunch_length
from xibs.inputs import BeamParameters, OpticsParameters

warnings.filterwarnings("ignore")  # scipy integration routines might warn

###############################################################################
# Let's start by defining the line and particle information, as well as some
# parameters for later use:

line_file = "lines/chrom-corr_DR.newlattice_2GHz.json"
harmonic_number = 2852
rf_voltage = 4.5  # in MV
energy_loss = 0  # let's pretend
bunch_intensity = 4.4e9
sigma_z = 1.58e-3
nemitt_x = 5.6644e-07
nemitt_y = 3.7033e-09
n_part = int(5e3)

###############################################################################
# Setting up line and particles
# -----------------------------
# Let's start by loading the `xtrack.Line`, activating RF cavities and 
# generating a matched Gaussian bunch:

line = xt.Line.from_json(line_file)
line.build_tracker(extra_headers=["#define XTRACK_MULTIPOLE_NO_SYNRAD"])

# ----- Power accelerating cavities ----- #
cavities = [element for element in line.elements if isinstance(element, xt.Cavity)]
for cavity in cavities:
    cavity.lag = 180

p0 = xp.Particles(mass0=xp.ELECTRON_MASS_EV, q0=1, p0c=2.86e9)
line.particle_ref = p0
twiss = line.twiss()

particles = xp.generate_matched_gaussian_bunch(
    num_particles=n_part,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    particle_ref=p0,
    line=line,
)

###############################################################################
# To have a look at the generated particles (see the `xsuite user guide 
# <https://xsuite.readthedocs.io/en/latest/particlesmanip.html>`_ for more)

fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(8, 11))

axx.plot(1e6 * particles.x, 1e5 * particles.px, ".", ms=3)
axy.plot(1e6 * particles.y, 1e6 * particles.py, ".", ms=3)
axz.plot(1e3 * particles.zeta, 1e3 * particles.delta, ".", ms=3)

axx.set_xlabel(r"$x$ [$\mu$m]")
axx.set_ylabel(r"$p_x$ [$10^{-5}$]")
axy.set_xlabel(r"$y$ [$\mu$m]")
axy.set_ylabel(r"$p_y$ [$10^{-3}$]")
axz.set_xlabel(r"$z$ [$10^{-3}$]")
axz.set_ylabel(r"$\delta$ [$10^{-3}$]")

plt.tight_layout()
plt.show()

###############################################################################
# We can compute initial (geometrical) emittances as well as the bunch length
# from the `xpart.Particles` object:

sig_x = np.std(particles.x[particles.state > 0])  # horizontal stdev
sig_y = np.std(particles.y[particles.state > 0])  # vertical stdev
sig_delta = np.std(particles.delta[particles.state > 0])  # momentum spread

# Compute horizontal & vertical geometric emittances as well as the bunch length, all in [m]
geom_epsx = (sig_x**2 - (twiss["dx"][0] * sig_delta) ** 2) / twiss["betx"][0]
geom_epsy = sig_y**2 / twiss["bety"][0]
bunch_l = np.std(particles.zeta[particles.state > 0])

###############################################################################
# Computing Elliptic Integrals and IBS Growth Rates
# -------------------------------------------------
# Let us initiate the `~.xibs.analytical.NagaitsevIBS` class. It is fairly
# simple and is done from both the beam parameters of the particles and the
# optics parameters of the line. For each of these a specific `dataclass`
# is provided in the `~.xibs.inputs` module.

beam_params = BeamParameters(particles)
optics = OpticsParameters(twiss)
IBS = NagaitsevIBS(beam_params, optics)

###############################################################################
# As a first step, all calculations in rely on the computing of Nagaitsev integrals
# :cite:p:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`, which is done with
# a dedicated function. These are returned, but also stored internally in the **IBS**
# object and will be updated internally each time they are computed.

integrals = IBS.integrals(geom_epsx, geom_epsy, sig_delta)
print(integrals)
print(IBS.elliptic_integrals)

###############################################################################
# From these the **IBS** growth rates can be computed, which is again done by calling
# a dedicated function. If the integrals mentioned above have not been computed yet,
# an error will be raised for the user. Once again, these are returned but also stored
# internally and updated internally each time they are computed.

growth_rates = IBS.growth_rates(geom_epsx, geom_epsy, sig_delta, bunch_l)
print(growth_rates)
print(IBS.ibs_growth_rates)

###############################################################################
# Computing New Emittances from Growth Rates
# ------------------------------------------
# From these one can compute the emittances at the next time step. For the emittances
# at the next turn, once should use :math:`1 / f_{rev}` as the time step, which
# is the default value used if it is not provided.

new_geom_epsx, new_geom_epsy, new_sig_delta = IBS.emittance_evolution(
    geom_epsx, geom_epsy, sig_delta
)
print(f"Next time step Epsilon_x = {new_geom_epsx}")
print(f"Next time step Epsilon_y = {new_geom_epsy}")
print(f"Next time step Sigma_delta = {new_sig_delta}")

###############################################################################
# Analytical Evolution for Many Turns
# -----------------------------------
# One can then analytically look at the evolution through many turns by looping
# over this calculation. Let's do this for 1000 turns. We will also re-compute
# the Nagaitsev elliptic integrals and the IBS grwoth rates every 50 turns.
# The more frequent this update the more accurate the results will be, but the
# slower the simulation as it is the most compute-intensive process.

nturns = 1000  # number of turns to loop for
ibs_step = 50  # frequency at which to re-compute the growth rates in [turns]

# Set up a dataclass to store the results
@dataclass
class Records:
    """Dataclass to store (and update) important values through tracking."""

    epsilon_x: np.ndarray
    epsilon_y: np.ndarray
    sig_delta: np.ndarray
    bunch_length: np.ndarray


# Initialize the dataclass
turn_by_turn = Records(
    epsilon_x=np.zeros(nturns, dtype=float),
    epsilon_y=np.zeros(nturns, dtype=float),
    sig_delta=np.zeros(nturns, dtype=float),
    bunch_length=np.zeros(nturns, dtype=float),
)

# Store the initial values
turn_by_turn.bunch_length[0] = np.std(particles.zeta[particles.state > 0])
turn_by_turn.sig_delta[0] = new_sig_delta
turn_by_turn.epsilon_x[0] = (sig_x**2 - (twiss["dx"][0] * new_sig_delta) ** 2) / twiss["betx"][0]
turn_by_turn.epsilon_y[0] = sig_y**2 / twiss["bety"][0]

# We loop here now
for turn in range(1, nturns):
    # ----- Potentially re-compute the Nagaitsev integrals and growth rates ----- #
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"Turn {turn}: re-computing the Nagaitsev integrals and growth rates")
        # We compute from values at the previous turn
        IBS.integrals(
            turn_by_turn.epsilon_x[turn - 1],
            turn_by_turn.epsilon_y[turn - 1],
            turn_by_turn.sig_delta[turn - 1],
        )
        IBS.growth_rates(
            turn_by_turn.epsilon_x[turn - 1],
            turn_by_turn.epsilon_y[turn - 1],
            turn_by_turn.sig_delta[turn - 1],
            turn_by_turn.bunch_length[turn - 1],
        )

    # ----- Compute the new emittances ----- #
    new_emit_x, new_emit_y, new_sig_delta = IBS.emittance_evolution(
        geom_epsx=turn_by_turn.epsilon_x[turn - 1],
        geom_epsy=turn_by_turn.epsilon_y[turn - 1],
        sigma_delta=turn_by_turn.sig_delta[turn - 1],
        # dt = 1.0 / IBS.optics.revolution_frequency,  # default value
    )

    # compute bunch length analytically as the Particles object hasn't changed
    sigma_e = new_sig_delta * IBS.beam_parameters.beta_rel**2
    bunch_l = bunch_length(
        optics.circumference,
        harmonic_number,
        beam_params.total_energy_GeV,
        optics.slip_factor,
        sigma_e,
        beam_params.beta_rel,
        rf_voltage * 1e-3,
        energy_loss,
        beam_params.particle_charge,
    )

    # ----- Update the records with the new values ----- #
    turn_by_turn.bunch_length[turn] = bunch_l
    turn_by_turn.sig_delta[turn] = new_sig_delta
    turn_by_turn.epsilon_x[turn] = new_emit_x
    turn_by_turn.epsilon_y[turn] = new_emit_y

###############################################################################
# Feel free to run this simulation for more turns, or with a different frequency
# of the IBS growth rates re-computation. After this is done running, we can plot
# the evolutions across the turns:

figure, (epsx, epsy, sigdelta) = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

epsx.plot(1e10 * turn_by_turn.epsilon_x)
epsy.plot(1e13 * turn_by_turn.epsilon_y)
sigdelta.plot(1e3 * turn_by_turn.sig_delta)

epsx.set_ylabel(r"$\varepsilon_x$ [$10^{-10}$m]")
epsy.set_ylabel(r"$\varepsilon_y$ [$10^{-13}$m]")
sigdelta.set_ylabel(r"$\sigma_{\delta}$ [$10^{-3}$]")
sigdelta.set_xlabel("Turn Number")

plt.tight_layout()
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `~xibs.analytical`: `~.xibs.analytical.NagaitsevIBS`
#    - `~xibs.inputs`: `~xibs.inputs.BeamParameters`, `~xibs.inputs.OpticsParameters`

###############################################################################
# TODO: remove below when this is resolved
# I am comparing to the old code (xibs._old_michail) now and finding some (small)
# differences. For instance:
# | ----------------- | ------------------ |
# | OLD CODE, IBS.Ixx | NEW CODE, IBS.ibs_growth_rates.Tx |
# | 356.725986354935  | 356.72598635530807  |
# | ----------------- | ------------------ |
# | OLD CODE, IBS.Iyy | NEW CODE, IBS.ibs_growth_rates.Ty |
# | 101.02642760866847 | 101.02642760877411 |
# | ----------------- | ------------------ |
# | OLD CODE, IBS.Ipp | NEW CODE, IBS.ibs_growth_rates.Tz |
# | 86.60401587714175 | 86.60401587723233 |
# | ----------------- | ------------------ |
# This may be due to the numba compiling of the iterative_RD method in the new
# code, that in turn leads to a different numerical precision than the python
# one in Michail's code? Literally same code but one is @jit()-ed
# Worthy to note that when doing a np.isclose() comparison, the assertions
# resolve to `True`. I will check without the jit() decorator and report below
# 
# Without the JIT decorator, the results are:
# | ----------------- | ------------------ |
# | OLD CODE, IBS.Ixx | NEW CODE, IBS.ibs_growth_rates.Tx |
# | 356.725986354935  | 356.72598635530807  |
# | ----------------- | ------------------ |
# | OLD CODE, IBS.Iyy | NEW CODE, IBS.ibs_growth_rates.Ty |
# | 101.02642760866847 | 101.02642760877411 |
# | ----------------- | ------------------ |
# | OLD CODE, IBS.Ipp | NEW CODE, IBS.ibs_growth_rates.Tz |
# | 86.60401587714175 | 86.60401587723233 |
# | ----------------- | ------------------ |
# 
# WE ARE LEARNING. Discrepancy is still here and still the same,
# the numba JIT is not responsible for it.
# 
# 
# OH MY GOD I WAS CALLING THE `IBS.emittance_evolution` function
# with the wrong argument for `dt` because I copy-pasted from how
# to call the old codes I'm benchmarking against. Like is good now
