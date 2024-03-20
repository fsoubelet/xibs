import warnings
import logging
import sys
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


plt.rcParams.update({"savefig.dpi": 300})

line_file = "lines/chrom-corr_DR.newlattice_2GHz.json"
bunch_intensity = int(4.5e9)
n_part = int(1.5e3)
sigma_z = 1.58e-3
nemitt_x = 5.66e-7
nemitt_y = 3.7e-9





line = xt.Line.from_json(line_file)
context = xo.ContextCpu(omp_num_threads="auto")
line.particle_ref = xp.Particles(mass0=xp.ELECTRON_MASS_EV, q0=1, p0c=2.86e9)

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
particles3 = particles.copy()






geom_epsx = _geom_epsx(particles, twiss.betx[0], twiss.dx[0])
geom_epsy = _geom_epsy(particles, twiss.bety[0], twiss.dy[0])
bunch_l = _bunch_length(particles)
sig_delta = _sigma_delta(particles)






logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stdout,
    format="[%(asctime)s] [%(levelname)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
)






AUTO_PERCENT = 8e-2
beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
opticsparams = OpticsParameters.from_line(line)
IBS = KineticKickIBS(beamparams, opticsparams)
AUTO_IBS = KineticKickIBS(beamparams, opticsparams, auto_recompute_coefficients_percent=AUTO_PERCENT)
MIX_IBS = KineticKickIBS(beamparams, opticsparams, auto_recompute_coefficients_percent=AUTO_PERCENT)






IBS.compute_kick_coefficients(particles2)
AUTO_IBS.compute_kick_coefficients(particles2)
MIX_IBS.compute_kick_coefficients(particles3)







IBS.apply_ibs_kick(particles2)
new_geom_epsx = _geom_epsx(particles2, twiss.betx[0], twiss.dx[0])
new_geom_epsy = _geom_epsy(particles2, twiss.bety[0], twiss.dy[0])
new_sig_delta = _sigma_delta(particles2)
new_bunch_length = _bunch_length(particles2)


print(f"Geom. epsx: {geom_epsx:.2e} -> {new_geom_epsx:.2e} | ({_percent_change(geom_epsx, new_geom_epsx):.2e}% change)")
print(f"Geom. epsy: {geom_epsy:.2e} -> {new_geom_epsy:.2e} | ({_percent_change(geom_epsy, new_geom_epsy):.2e}% change)")
print(f"Sigma delta: {sig_delta:.2e} -> {new_sig_delta:.2e} | ({_percent_change(sig_delta, new_sig_delta):.2e}% change)")
print(f"Bunch length: {bunch_l:.4e} -> {new_bunch_length:.4e} | ({_percent_change(bunch_l, new_bunch_length):.2e}% change)")




particles2 = particles.copy()  # reset that
particles3 = particles.copy()  # reset that




nturns = 1000  # number of turns to loop for
ibs_step = 50  # frequency at which to re-compute coefficients in [turns]
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






regular = Records.init_zeroes(nturns)
regular.update_at_turn(0, particles, twiss)

auto = Records.init_zeroes(nturns)
auto.update_at_turn(0, particles, twiss)

mix = Records.init_zeroes(nturns)
mix.update_at_turn(0, particles, twiss)



auto_recomputes = np.zeros(nturns, dtype=float)
mix_recomputes = np.zeros(nturns, dtype=float)
fixed_recomputes = np.zeros(nturns, dtype=float)




# ----- We loop here now ----- #
for turn in range(1, nturns):
    n_recomputes_for_auto = AUTO_IBS._number_of_coefficients_computations
    n_recomputes_for_mix = MIX_IBS._number_of_coefficients_computations
    # ----- Potentially re-compute the IBS growth rates and kick coefficients ----- #
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"Turn {turn:d}: re-computing diffusion and friction terms")
        # Compute kick coefficients from the particle distribution at this moment
        IBS.compute_kick_coefficients(particles)
        MIX_IBS.compute_kick_coefficients(particles3)

    # ----- Manually Apply IBS Kick and Track Turn ----- #
    IBS.apply_ibs_kick(particles)
    AUTO_IBS.apply_ibs_kick(particles2)
    MIX_IBS.apply_ibs_kick(particles3)
    line.track(particles, num_turns=1)
    line.track(particles2, num_turns=1)
    line.track(particles3, num_turns=1)

    # ----- Update records for tracked particles ----- #
    regular.update_at_turn(turn, particles, twiss)
    auto.update_at_turn(turn, particles2, twiss)
    mix.update_at_turn(turn, particles3, twiss)

    # ----- Check if the rates were auto-recomputed ----- #
    if AUTO_IBS._number_of_coefficients_computations > n_recomputes_for_auto:
        print(f"At turn {turn} - Auto re-computed coefficients")
        auto_recomputes[turn - 1] = 1
    if MIX_IBS._number_of_coefficients_computations > n_recomputes_for_mix:
        print(f"At turn {turn} - Mix re-computed coefficients")
        mix_recomputes[turn - 1] = 1


for turn in turns:
    if (turn % ibs_step == 0) or (turn == 1):
        fixed_recomputes[turn - 1] = 1


where_auto_recomputes = np.flatnonzero(auto_recomputes)
where_mix_recomputes = np.flatnonzero(mix_recomputes)
where_fixed_recomputes = np.flatnonzero(fixed_recomputes)



print(f"Fixed re-computes: {IBS._number_of_coefficients_computations}")
print(f"Auto re-computes: {AUTO_IBS._number_of_coefficients_computations}")
print(
    f"Mix re-computes: {MIX_IBS._number_of_coefficients_computations}, "
    f"(on its own: {MIX_IBS._number_of_coefficients_computations - IBS._number_of_coefficients_computations})"
)





AVG_TURNS = 3  # for rolling average plotting
# ----- Plotting the results ----- #

fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(13, 7))

# Add vertical lines at recompute times (first so they show up in the background)
for axis in axs.values():
    for turn in where_auto_recomputes:
        axis.axvline(turn, color="C1", linestyle="-", alpha=0.2)
    for turn in where_mix_recomputes:
        axis.axvline(turn, color="C2", linestyle="-", alpha=0.2)
    for turn in where_fixed_recomputes:
        axis.axvline(turn, color="gray", linestyle="--", lw=1, alpha=0.7)


axs["epsx"].plot(turns, 1e10 * pd.Series(regular.epsilon_x).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=2, label=f"Fixed ({ibs_step}) turns")
axs["epsy"].plot(turns, 1e13 * pd.Series(regular.epsilon_y).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=2, label=f"Fixed ({ibs_step}) turns")
axs["sigd"].plot(turns, 1e3 * pd.Series(regular.sigma_delta).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=2, label=f"Fixed ({ibs_step}) turns")
axs["bl"].plot(turns, 1e3 * pd.Series(regular.bunch_length).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=2, label=f"Fixed ({ibs_step}) turns")

axs["epsx"].plot(turns, 1e10 * pd.Series(auto.epsilon_x).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=1.9, label=f"Auto ({AUTO_PERCENT:.0e}% change)")
axs["epsy"].plot(turns, 1e13 * pd.Series(auto.epsilon_y).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=1.9, label=f"Auto ({AUTO_PERCENT:.0e}% change)")
axs["sigd"].plot(turns, 1e3 * pd.Series(auto.sigma_delta).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=1.9, label=f"Auto ({AUTO_PERCENT:.0e}% change)")
axs["bl"].plot(turns, 1e3 * pd.Series(auto.bunch_length).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=1.9, label=f"Auto ({AUTO_PERCENT:.0e}% change)")

axs["epsx"].plot(turns, 1e10 * pd.Series(mix.epsilon_x).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=1.9, label="Mix")
axs["epsy"].plot(turns, 1e13 * pd.Series(mix.epsilon_y).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=1.9, label="Mix")
axs["sigd"].plot(turns, 1e3 * pd.Series(mix.sigma_delta).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=1.9, label="Mix")
axs["bl"].plot(turns, 1e3 * pd.Series(mix.bunch_length).rolling(AVG_TURNS, closed="both", min_periods=1).mean(), lw=1.9, label="Mix")

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

plt.legend(title="Recompute Rates")
plt.tight_layout()
plt.show()






# ----- Plotting the percent changes ----- #

fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(13, 7))

# Add vertical lines at recompute times (first so they show up in the background)
for axis in axs.values():
    for turn in where_auto_recomputes:
        axis.axvline(turn, color="C1", linestyle="-", alpha=0.3)
    for turn in where_mix_recomputes:
        axis.axvline(turn, color="C2", linestyle="-", alpha=0.2)
    for turn in where_fixed_recomputes:
        axis.axvline(turn, color="gray", linestyle="--", lw=1, alpha=0.7)


axs["epsx"].plot(turns, 1e2 * pd.Series(regular.epsilon_x).pct_change(), lw=1, label=f"Fixed ({ibs_step}) turns")
axs["epsy"].plot(turns, 1e2 * pd.Series(regular.epsilon_y).pct_change(), lw=1, label=f"Fixed ({ibs_step}) turns")
axs["sigd"].plot(turns, 1e2 * pd.Series(regular.sigma_delta).pct_change(), lw=1, label=f"Fixed ({ibs_step}) turns")
axs["bl"].plot(turns, 1e2 * pd.Series(regular.bunch_length).pct_change(), lw=1, label=f"Fixed ({ibs_step}) turns")

axs["epsx"].plot(turns, 1e2 * pd.Series(auto.epsilon_x).pct_change(), lw=1, label=f"Auto ({AUTO_PERCENT:.0e}% change)")
axs["epsy"].plot(turns, 1e2 * pd.Series(auto.epsilon_y).pct_change(), lw=1, label=f"Auto ({AUTO_PERCENT:.0e}% change)")
axs["sigd"].plot(turns, 1e2 * pd.Series(auto.sigma_delta).pct_change(), lw=1, label=f"Auto ({AUTO_PERCENT:.0e}% change)")
axs["bl"].plot(turns, 1e2 * pd.Series(auto.bunch_length).pct_change(), lw=1, label=f"Auto ({AUTO_PERCENT:.0e}% change)")

axs["epsx"].plot(turns, 1e2 * pd.Series(mix.epsilon_x).pct_change(), lw=1, label="Mix")
axs["epsy"].plot(turns, 1e2 * pd.Series(mix.epsilon_y).pct_change(), lw=1, label="Mix")
axs["sigd"].plot(turns, 1e2 * pd.Series(mix.sigma_delta).pct_change(), lw=1, label="Mix")
axs["bl"].plot(turns, 1e2 * pd.Series(mix.bunch_length).pct_change(), lw=1, label="Mix")


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

plt.legend(title="Recompute Rates")
plt.tight_layout()
plt.show()
