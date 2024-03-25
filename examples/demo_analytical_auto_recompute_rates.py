"""

.. _demo-analytical-auto-growth-rates:

=====================================
Automatic Growth Rates Re-Computation
=====================================

This example shows how to use the the auto recomputing functionality for growth rates
in the analytical classes. To follow this example please first have a look at the 
:ref:`Analytical Nagaitsev example <demo-analytical-nagaitsev>` as well as the relevant
:ref:`FAQ section <xibs-faq-auto-recompute-growth-rates>`.

This script will showcase the functionality by following an identical flow to the 
Nagaitsev example linked above and expanding on the relevant parts. Demonstration will
be done using the SPS top protons line.
"""
# sphinx_gallery_thumbnail_number = 1
import logging

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import xpart as xp
import xtrack as xt

from xibs.analytical import NagaitsevIBS
from xibs.formulary import _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta, _percent_change
from xibs.inputs import BeamParameters, OpticsParameters

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] [%(levelname)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
)
# plt.rcParams.update({"savefig.dpi": 300})
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
# The beam parameters are very exaggerated to enhance IBS effects
bunch_intensity = 5e11
sigma_z = 10e-2  # very pushed
nemitt_x = 1e-6  # very pushed
nemitt_y = 1e-6  # very pushed
n_part = int(5e3)

###############################################################################
# Setting up line and particles
# -----------------------------
# Let's start by loading the `xtrack.Line`, activating RF cavities and
# generating a matched Gaussian bunch:

line = xt.Line.from_json(line_file)
line.build_tracker()
line.optimize_for_tracking()
twiss = line.twiss(method="4d")

# ----- Power accelerating cavities ----- #

rf_voltage = 4  # in MV
rf_frequency = 200e6
harmonic_number = 4653
cavity_name = "actcse.31632"
cavity_lag = 180
line[cavity_name].voltage = rf_voltage * 1e6  # to be given in [V] here
line[cavity_name].lag = cavity_lag
line[cavity_name].frequency = rf_frequency

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
# We can compute initial (geometrical) emittances as well as the bunch length
# from the `xtrack.Particles` object:

geom_epsx = _geom_epsx(particles, twiss.betx[0], twiss.dx[0])
geom_epsy = _geom_epsy(particles, twiss.bety[0], twiss.dy[0])
bunch_l = _bunch_length(particles)
sig_delta = _sigma_delta(particles)

###############################################################################
# Instantiate Analytical IBS Objects
# -------------------------------------------------
# Let us instantiate three instances of the `~.xibs.analytical.NagaitsevIBS` class,
# where each will be used for a different updating behaviour of the growth rates.

beam_params = BeamParameters(particles)
optics = OpticsParameters(twiss)

IBS = NagaitsevIBS(beam_params, optics)  # will recompute rates at a given frequency
AUTO_IBS = NagaitsevIBS(beam_params, optics)  # will auto-recompute rates only

###############################################################################
# Let's get them growth rates and see the evolution of the transverse emittances,
# :math:`\sigma_{\delta}` and bunch length. We can then have a look at the relative
# change from one second to the next.

IBS.growth_rates(geom_epsx, geom_epsy, sig_delta, bunch_l)
AUTO_IBS.growth_rates(geom_epsx, geom_epsy, sig_delta, bunch_l)

new_geom_epsx, new_geom_epsy, new_sig_delta, new_bunch_length = IBS.emittance_evolution(
    geom_epsx, geom_epsy, sig_delta, bunch_l
)

# Let's see the relative change
print(f"Geom. epsx: {geom_epsx:.2e} -> {new_geom_epsx:.2e} | ({_percent_change(geom_epsx, new_geom_epsx):.2e}% change)")
print(f"Geom. epsy: {geom_epsy:.2e} -> {new_geom_epsy:.2e} | ({_percent_change(geom_epsy, new_geom_epsy):.2e}% change)")
print(f"Sigma delta: {sig_delta:.2e} -> {new_sig_delta:.2e} | ({_percent_change(sig_delta, new_sig_delta):.2e}% change)")
print(f"Bunch length: {bunch_l:.2e} -> {new_bunch_length:.2e} | ({_percent_change(bunch_l, new_bunch_length):.2e}% change)")

###############################################################################
# Preparing for Simulation of Evolution
# -------------------------------------
# We will loop over time steps (seconds in this case) for 3 hours equivalent time,
# and specify the auto-recomputing in these steps where relevant. Let's set up the
# utilities we will need for this:

nsecs = 5 * 3_600  # that's 5h
ibs_step = 10 * 60  # fixed interval for rates recomputing
seconds = np.linspace(0, nsecs, nsecs, dtype=int)

# This is the threshold that would trigger the auto-recomping of the growth rates.
AUTO_PERCENT = 15  # threshold: 15% change from last growth rates update

# Set up a dataclass to store the results
@dataclass
class Records:
    """Dataclass to store (and update) important values through tracking."""

    epsilon_x: np.ndarray  # geometric horizontal emittance in [m]
    epsilon_y: np.ndarray  # geometric vertical emittance in [m]
    sig_delta: np.ndarray  # momentum spread
    bunch_length: np.ndarray  # bunch length in [m]
    epsx_rel_to_last_ref: np.ndarray  # relative change of epsx to last reference
    epsy_rel_to_last_ref: np.ndarray  # relative change of epsy to last reference
    sigd_rel_to_last_ref: np.ndarray  # relative change of sigd to last reference
    bl_rel_to_last_ref: np.ndarray  # relative change of bl to last reference


    @classmethod
    def init_zeroes(cls, n_turns: int):
        return cls(
            epsilon_x=np.zeros(n_turns, dtype=float),
            epsilon_y=np.zeros(n_turns, dtype=float),
            sig_delta=np.zeros(n_turns, dtype=float),
            bunch_length=np.zeros(n_turns, dtype=float),
            epsx_rel_to_last_ref=np.zeros(n_turns, dtype=float),
            epsy_rel_to_last_ref=np.zeros(n_turns, dtype=float),
            sigd_rel_to_last_ref=np.zeros(n_turns, dtype=float),
            bl_rel_to_last_ref=np.zeros(n_turns, dtype=float),
        )

    def update_at_turn(self, turn: int, epsx: float, epsy: float, sigd: float, bl: float, epsxrel: float = 0.0, epsyrel: float = 0.0, sigdrel: float = 0.0, blrel: float = 0.0):
        """Works for turns / seconds, just needs the correct index to store in."""
        self.epsilon_x[turn] = epsx
        self.epsilon_y[turn] = epsy
        self.sig_delta[turn] = sigd
        self.bunch_length[turn] = bl
        self.epsx_rel_to_last_ref[turn] = epsxrel
        self.epsy_rel_to_last_ref[turn] = epsyrel
        self.sigd_rel_to_last_ref[turn] = sigdrel
        self.bl_rel_to_last_ref[turn] = blrel


# Initialize the dataclass & store the initial values
regular = Records.init_zeroes(nsecs)
auto = Records.init_zeroes(nsecs)

regular.update_at_turn(0, geom_epsx, geom_epsy, sig_delta, bunch_l)
auto.update_at_turn(0, geom_epsx, geom_epsy, sig_delta, bunch_l)

# These arrays we will use to see when the auto-recomputing kicks in
auto_recomputes = np.zeros(nsecs, dtype=float)
fixed_recomputes = np.zeros(nsecs, dtype=float)

###############################################################################
# Analytical Evolution Over Time
# ------------------------------
# Let's now loop and specify the auto-recomputing where relevant.

for sec in range(1, nsecs):
    # This is not necessary, just for showcasing in this tutorial
    n_recomputes_for_auto = AUTO_IBS._number_of_growth_rates_computations

    # ----- Potentially re-compute the IBS growth rates ----- #
    if (sec % ibs_step == 0) or (sec == 1):
        print(f"At {sec}s: Fixed interval recomputing of the growth rates")
        # Both below always re-compute the rates every 'ibs_step' seconds
        IBS.growth_rates(
            regular.epsilon_x[sec - 1],
            regular.epsilon_y[sec - 1],
            regular.sig_delta[sec - 1],
            regular.bunch_length[sec - 1],
        )

    # ----- Compute the new emittances ----- #
    new_emit_x, new_emit_y, new_sig_delta, new_bunch_length = IBS.emittance_evolution(
        epsx=regular.epsilon_x[sec - 1],
        epsy=regular.epsilon_y[sec - 1],
        sigma_delta=regular.sig_delta[sec - 1],
        bunch_length=regular.bunch_length[sec - 1],
        dt=1.0,  # get at next second
    )
    regular.update_at_turn(sec,
        new_emit_x,
        new_emit_y,
        new_sig_delta,
        new_bunch_length,
        _percent_change(IBS._refs.epsx, new_emit_x),
        _percent_change(IBS._refs.epsy, new_emit_y),
        _percent_change(IBS._refs.sigma_delta, new_sig_delta),
        _percent_change(IBS._refs.bunch_length, new_bunch_length),
    )

    # ----- Potentially auto-update growth rates and compute new emittances ----- #
    # The AUTO_IBS is given our threshold for auto-recomputing and will decide to update
    # its growth rates itself if any of the quantities change by more than 15%.
    anew_emit_x, anew_emit_y, anew_sig_delta, anew_bunch_length = AUTO_IBS.emittance_evolution(
        epsx=auto.epsilon_x[sec - 1],
        epsy=auto.epsilon_y[sec - 1],
        sigma_delta=auto.sig_delta[sec - 1],
        bunch_length=auto.bunch_length[sec - 1],
        dt=1.0,  # get at next second
        auto_recompute_rates_percent=AUTO_PERCENT,
    )
    auto.update_at_turn(
        sec,
        anew_emit_x,
        anew_emit_y,
        anew_sig_delta,
        anew_bunch_length,
        _percent_change(AUTO_IBS._refs.epsx, anew_emit_x),
        _percent_change(AUTO_IBS._refs.epsy, anew_emit_y),
        _percent_change(AUTO_IBS._refs.sigma_delta, anew_sig_delta),
        _percent_change(AUTO_IBS._refs.bunch_length, anew_bunch_length),
    )

    # ----- Check if the rates were auto-recomputed ----- #
    # This is also not necessary, just for showcasing in this tutorial
    if AUTO_IBS._number_of_growth_rates_computations > n_recomputes_for_auto:
        print(f"At {sec}s - Auto re-computed growth rates")
        auto_recomputes[sec] = 1


# Here we also aggregate the fixed recomputes
for sec in seconds:
    if (sec % ibs_step == 0) or (sec == 1):
        fixed_recomputes[sec - 1] = 1

# And these are simply 1D arrays of the seconds at which re-computing happened
where_auto_recomputes = np.flatnonzero(auto_recomputes)
where_fixed_recomputes = np.flatnonzero(fixed_recomputes)

print(f"Fixed re-computes: {IBS._number_of_growth_rates_computations}")
print(f"Auto re-computes: {AUTO_IBS._number_of_growth_rates_computations}")

###############################################################################
# Feel free to run this simulation with different parameters, but beware that the
# lower the auto-recompute threshold, the more likely it is that growth rates will
# be re-computed at every turn, which can be very lengthy. Let's first have a look
# at the evolution of emittances over time:

fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(13, 8.5))

# We will add vertical lines at the times where recomputing of the growth 
# rates happened (do this first so they show up in the background)
for axis in axs.values():
    for sec in where_fixed_recomputes:
        axis.axvline(sec / 3600, color="C0", linestyle="--", lw=1, alpha=0.5)
    for sec in where_auto_recomputes:
        axis.axvline(sec / 3600, color="C1", linestyle="-", alpha=0.035)

axs["epsx"].plot(seconds / 3600, 1e9 * regular.epsilon_x, lw=2, label=f"Fixed ({int(ibs_step / 60)} mins)")
axs["epsy"].plot(seconds / 3600, 1e9 * regular.epsilon_y, lw=2, label=f"Fixed ({int(ibs_step / 60)} mins)")
axs["sigd"].plot(seconds / 3600, 1e4 * regular.sig_delta, lw=2, label=f"Fixed ({int(ibs_step / 60)} mins)")
axs["bl"].plot(seconds / 3600, 1e2 * regular.bunch_length, lw=2, label=f"Fixed ({int(ibs_step / 60)} mins)")

axs["epsx"].plot(seconds / 3600, 1e9 * auto.epsilon_x, lw=2, label=f"Auto ({AUTO_PERCENT:.0f}% change)")
axs["epsy"].plot(seconds / 3600, 1e9 * auto.epsilon_y, lw=2, label=f"Auto ({AUTO_PERCENT:.0f}% change)")
axs["sigd"].plot(seconds / 3600, 1e4 * auto.sig_delta, lw=2, label=f"Auto ({AUTO_PERCENT:.0f}% change)")
axs["bl"].plot(seconds / 3600, 1e2 * auto.bunch_length, lw=2, label=f"Auto ({AUTO_PERCENT:.0f}% change)")

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

fig.align_ylabels((axs["epsx"], axs["sigd"]))
fig.align_ylabels((axs["epsy"], axs["bl"]))
fig.suptitle("Analytical Evolution of Emittances from IBS\nSPS Top Protons")

plt.legend(title="Recompute Rates")
plt.tight_layout()
plt.show()


###############################################################################
# Does this make sense?
# Yes, it does once we keep in mind that the formula for the evolution of these
# properties from IBS is an exponential that depends on the value at the previous
# step, and the growth rate for the given plane. As long as the growth rate is
# not re-computed, then the relative change from one time step to the next will
# be the same. We can confirm this by plotting the relative change:

fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(13, 8.5))

# We will add vertical lines at the times where recomputing of the growth 
# rates happened (do this first so they show up in the background)
for axis in axs.values():
    for sec in where_fixed_recomputes:
        axis.axvline(sec / 3600, color="C0", linestyle="--", lw=1, alpha=0.5)
    for sec in where_auto_recomputes:
        axis.axvline(sec / 3600, color="C1", linestyle="-", alpha=0.035)

axs["epsx"].plot(seconds / 3600, 1e2 * regular.epsx_rel_to_last_ref, lw=2, label=f"Fixed ({int(ibs_step / 60)} mins)")
axs["epsy"].plot(seconds / 3600, 1e2 * regular.epsy_rel_to_last_ref, lw=2, label=f"Fixed ({int(ibs_step / 60)} mins)")
axs["sigd"].plot(seconds / 3600, 1e2 * regular.sigd_rel_to_last_ref, lw=2, label=f"Fixed ({int(ibs_step / 60)} mins)")
axs["bl"].plot(seconds / 3600, 1e2 * regular.bl_rel_to_last_ref, lw=2, label=f"Fixed ({int(ibs_step / 60)} mins)")

axs["epsx"].plot(seconds / 3600, 1e2 * auto.epsx_rel_to_last_ref, lw=2, label=f"Auto ({AUTO_PERCENT:.0f}% change)")
axs["epsy"].plot(seconds / 3600, 1e2 * auto.epsy_rel_to_last_ref, lw=2, label=f"Auto ({AUTO_PERCENT:.0f}% change)")
axs["sigd"].plot(seconds / 3600, 1e2 * auto.sigd_rel_to_last_ref, lw=2, label=f"Auto ({AUTO_PERCENT:.0f}% change)")
axs["bl"].plot(seconds / 3600, 1e2 * auto.bl_rel_to_last_ref, lw=2, label=f"Auto ({AUTO_PERCENT:.0f}% change)")

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
    axis.axhline(AUTO_PERCENT, color="black", linestyle="--", alpha=0.5, label="Threshold")
    axis.yaxis.set_major_locator(plt.MaxNLocator(3))
    axis.set_yscale("log")

fig.align_ylabels((axs["epsx"], axs["sigd"]))
fig.align_ylabels((axs["epsy"], axs["bl"]))
fig.suptitle("Percent change from values at previous growth rate update\nSPS Top Protons")

plt.legend(title="Recompute Rates")
plt.tight_layout()
plt.show()


###############################################################################
# We can see from this plot that the relative change for the "fixed" interval
# scenario is constant between re-computes of the growth rates. Because we pushed
# the beam parameters to the extreme, the emittances growth "too fast" from the
# start to the first recompute of the rates 10 minutes in.
#
# The "auto-only" recomputing scenario initially recomputes growth rates at each
# time step as the relative change of emittances is above the given threshold, mostly
# since we started with very exaggerated beam parameters. At some point, around 40 mins
# in, our bunch has expanded significantly the effect of IBS decreased, and the relative
# change of the emittances is low enough that the rates are not updated anymore. This
# means these growth rates are used until the end and the changes are too high for the
# rest of the simulation.
#
# The "mix" scenario brings the best of both worlds. It initially also recomputes the
# growth rates at every step, which makes sure emittances have the proper evolution
# during the most sensitive interval right at the start. However as a an update is
# forced at fixed intervals, the growth rates stay relatively up-to-date until the
# end of the simulation.
#
# Takeaways
# ---------
# One can see the difference made by the auto-recomputing, especially in the horizontal
# plane, where a clear gap to the "fixed interval" scenario is observed for the final 
# values reached (green to blue curves), despite an asymptotic behaviour.
# The "auto-only" scenario shows the potential of the auto-recomputing, but also the
# importance of choosing an appropriate threshold value. A value too high will lead
# to the rates stopping their updates too early and unrealistic evolutions, while a
# value too low will lead to the rates re-computing at every step for too long, which
# is computationally expensive and unnecessary as a regular interval does suffice.


#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `~xibs.analytical`: `~.xibs.analytical.NagaitsevIBS`, `~.xibs.analytical.NagaitsevIBS.growth_rates`, `~.xibs.analytical.NagaitsevIBS.integrals`, `~.xibs.analytical.NagaitsevIBS.emittance_evolution`
#    - `~xibs.inputs`: `~xibs.inputs.BeamParameters`, `~xibs.inputs.OpticsParameters`

###############################################################################
