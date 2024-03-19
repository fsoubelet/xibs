import warnings
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xpart as xp
import xtrack as xt

from xibs.analytical import NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters

warnings.simplefilter("ignore")  # for this tutorial's clarity


xibs_repo = Path(__file__).parent.parent.absolute()
filepath = xibs_repo / "tests" / "inputs" / "lines" / "lhc_top_protons.json"
line = xt.Line.from_json(filepath)


# Let's absolutely push the beam parameters
bunch_intensity = 1e14
bunch_length_m = 2.5e-2
sigma_delta = 1e-4
geom_epsx = 1e-9
geom_epsy = 1e-9





line.build_tracker()
line.optimize_for_tracking()
twiss = line.twiss(method="4d")




beam_params = BeamParameters.from_line(line, bunch_intensity)
optics = OpticsParameters(twiss)
IBS = NagaitsevIBS(beam_params, optics)  # recomputes at a given frequency
AUTO_IBS = NagaitsevIBS(beam_params, optics)  # only auto recomputes
MIX_IBS = NagaitsevIBS(beam_params, optics)  # does both


IBS.growth_rates(geom_epsx, geom_epsy, sigma_delta, bunch_length_m)
AUTO_IBS.growth_rates(geom_epsx, geom_epsy, sigma_delta, bunch_length_m)
MIX_IBS.growth_rates(geom_epsx, geom_epsy, sigma_delta, bunch_length_m)


nsecs = 5 * 3_600  # that's 5h
ibs_step = 10 * 60  # re-compute rates every 20min
seconds = np.linspace(0, nsecs, nsecs).astype(int)


# Set up a dataclass to store the results
@dataclass
class Records:
    """Dataclass to store (and update) important values through tracking."""

    epsilon_x: np.ndarray  # geometric horizontal emittance in [m]
    epsilon_y: np.ndarray  # geometric vertical emittance in [m]
    sig_delta: np.ndarray  # momentum spread
    bunch_length: np.ndarray  # bunch length in [m]

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
regular = Records.init_zeroes(nsecs)
regular.update_at_turn(0, geom_epsx, geom_epsy, sigma_delta, bunch_length_m)

auto = Records.init_zeroes(nsecs)
auto.update_at_turn(0, geom_epsx, geom_epsy, sigma_delta, bunch_length_m)

mix = Records.init_zeroes(nsecs)
mix.update_at_turn(0, geom_epsx, geom_epsy, sigma_delta, bunch_length_m)


auto_recomputes = np.zeros(nsecs, dtype=float)
mix_recomputes = np.zeros(nsecs, dtype=float)
fixed_recomputes = np.zeros(nsecs, dtype=float)


# ----- We loop here now ----- #

AUTO_PERCENT = 5e-3
for sec in range(1, nsecs):
    n_recomputes_for_auto = AUTO_IBS._number_of_growth_rates_computations
    n_recomputes_for_mix = MIX_IBS._number_of_growth_rates_computations
    # ----- Potentially re-compute the IBS growth rates ----- #
    if (sec % ibs_step == 0) or (sec == 1):
        print(f"At {sec}s: re-computing growth rates the regular way")
        # We compute from values at the previous turn
        IBS.growth_rates(
            regular.epsilon_x[sec - 1],
            regular.epsilon_y[sec - 1],
            regular.sig_delta[sec - 1],
            regular.bunch_length[sec - 1],
        )
        MIX_IBS.growth_rates(
            mix.epsilon_x[sec - 1],
            mix.epsilon_y[sec - 1],
            mix.sig_delta[sec - 1],
            mix.bunch_length[sec - 1],
        )

    # ----- Compute the new emittances ----- #
    new_emit_x, new_emit_y, new_sig_delta, new_bunch_length = IBS.emittance_evolution(
        epsx=regular.epsilon_x[sec - 1],
        epsy=regular.epsilon_y[sec - 1],
        sigma_delta=regular.sig_delta[sec - 1],
        bunch_length=regular.bunch_length[sec - 1],
        dt=1.0,  # get at next second
    )
    regular.update_at_turn(sec, new_emit_x, new_emit_y, new_sig_delta, new_bunch_length)

    # ----- Potentially auto-update growth rates and compute new emittances ----- #
    anew_emit_x, anew_emit_y, anew_sig_delta, anew_bunch_length = AUTO_IBS.emittance_evolution(
        epsx=auto.epsilon_x[sec - 1],
        epsy=auto.epsilon_y[sec - 1],
        sigma_delta=auto.sig_delta[sec - 1],
        bunch_length=auto.bunch_length[sec - 1],
        dt=1.0,  # get at next second
        auto_recompute_rates_percent=AUTO_PERCENT,
    )
    auto.update_at_turn(sec, anew_emit_x, anew_emit_y, anew_sig_delta, anew_bunch_length)

    # ----- Potentially auto-update growth rates and compute new emittances ----- #
    mnew_emit_x, mnew_emit_y, mnew_sig_delta, mnew_bunch_length = MIX_IBS.emittance_evolution(
        epsx=mix.epsilon_x[sec - 1],
        epsy=mix.epsilon_y[sec - 1],
        sigma_delta=mix.sig_delta[sec - 1],
        bunch_length=mix.bunch_length[sec - 1],
        dt=1.0,  # get at next second
        auto_recompute_rates_percent=AUTO_PERCENT,
    )
    mix.update_at_turn(sec, mnew_emit_x, mnew_emit_y, mnew_sig_delta, mnew_bunch_length)

    # ----- Check if the rates were auto-recomputed ----- #
    if AUTO_IBS._number_of_growth_rates_computations > n_recomputes_for_auto:
        print(f"At {sec}s - Auto re-computed growth rates")
        auto_recomputes[sec] = 1
    if MIX_IBS._number_of_growth_rates_computations > n_recomputes_for_mix:
        print(f"At {sec}s - Mix re-computed growth rates")
        mix_recomputes[sec] = 1




for sec in seconds:
    if (sec % ibs_step == 0) or (sec == 1):
        fixed_recomputes[sec - 1] = 1

where_auto_recomputes = np.flatnonzero(auto_recomputes)
where_mix_recomputes = np.flatnonzero(mix_recomputes)
where_fixed_recomputes = np.flatnonzero(fixed_recomputes)


print(f"Fixed re-computes: {IBS._number_of_growth_rates_computations}")
print(f"Auto re-computes: {AUTO_IBS._number_of_growth_rates_computations}")
print(
    f"Mix re-computes: {MIX_IBS._number_of_growth_rates_computations}, "
    f"(on its own: {MIX_IBS._number_of_growth_rates_computations - IBS._number_of_growth_rates_computations})"
)




# ----- Plotting the results ----- #

fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(13, 7))

# Add vertical lines at recompute times (first so they show up in the background)
for axis in axs.values():
    for sec in where_auto_recomputes:
        axis.axvline(sec / 3600, color="C1", linestyle="-", alpha=0.002)
    for sec in where_mix_recomputes:  # if auto recomputed this did too
        axis.axvline(sec / 3600, color="C2", linestyle="-", alpha=0.002)
    for sec in where_fixed_recomputes:
        axis.axvline(sec / 3600, color="gray", linestyle="--", lw=1, alpha=0.2)

axs["epsx"].plot(seconds / 3600, regular.epsilon_x, lw=2, label="Fixed")
axs["epsy"].plot(seconds / 3600, regular.epsilon_y, lw=2, label="Fixed")
axs["sigd"].plot(seconds / 3600, regular.sig_delta, lw=2, label="Fixed")
axs["bl"].plot(seconds / 3600, regular.bunch_length, lw=2, label="Fixed")

axs["epsx"].plot(seconds / 3600, auto.epsilon_x, lw=1.9, label="Auto")
axs["epsy"].plot(seconds / 3600, auto.epsilon_y, lw=1.9, label="Auto")
axs["sigd"].plot(seconds / 3600, auto.sig_delta, lw=1.9, label="Auto")
axs["bl"].plot(seconds / 3600, auto.bunch_length, lw=1.9, label="Auto")

axs["epsx"].plot(seconds / 3600, mix.epsilon_x, lw=1.7, label="Mix")
axs["epsy"].plot(seconds / 3600, mix.epsilon_y, lw=1.7, label="Mix")
axs["sigd"].plot(seconds / 3600, mix.sig_delta, lw=1.7, label="Mix")
axs["bl"].plot(seconds / 3600, mix.bunch_length, lw=1.7, label="Mix")

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
fig.suptitle("Analytical Evolution of Emittances from IBS - SPS Top Ions")

plt.legend(title="Recompute Rates")
plt.tight_layout()
plt.show()


# ----- Plotting the percent changes ----- #

fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(13, 7))

# Add vertical lines at recompute times (first so they show up in the background)
for axis in axs.values():
    for sec in where_auto_recomputes:
        axis.axvline(sec / 3600, color="C1", linestyle="-", alpha=0.002)
    for sec in where_mix_recomputes:  # if auto recomputed this did too
        axis.axvline(sec / 3600, color="C2", linestyle="-", alpha=0.002)
    for sec in where_fixed_recomputes:
        axis.axvline(sec / 3600, color="gray", linestyle="--", lw=1, alpha=0.2)

axs["epsx"].plot(seconds / 3600, 1e2 * pd.Series(regular.epsilon_x).pct_change(), lw=2, label="Fixed")
axs["epsy"].plot(seconds / 3600, 1e2 * pd.Series(regular.epsilon_y).pct_change(), lw=2, label="Fixed")
axs["sigd"].plot(seconds / 3600, 1e2 * pd.Series(regular.sig_delta).pct_change(), lw=2, label="Fixed")
axs["bl"].plot(seconds / 3600, 1e2 * pd.Series(regular.bunch_length).pct_change(), lw=2, label="Fixed")

axs["epsx"].plot(seconds / 3600, 1e2 * pd.Series(auto.epsilon_x).pct_change(), lw=2, label="Auto")
axs["epsy"].plot(seconds / 3600, 1e2 * pd.Series(auto.epsilon_y).pct_change(), lw=2, label="Auto")
axs["sigd"].plot(seconds / 3600, 1e2 * pd.Series(auto.sig_delta).pct_change(), lw=2, label="Auto")
axs["bl"].plot(seconds / 3600, 1e2 * pd.Series(auto.bunch_length).pct_change(), lw=2, label="Auto")

axs["epsx"].plot(seconds / 3600, 1e2 * pd.Series(mix.epsilon_x).pct_change(), lw=2, label="Mix")
axs["epsy"].plot(seconds / 3600, 1e2 * pd.Series(mix.epsilon_y).pct_change(), lw=2, label="Mix")
axs["sigd"].plot(seconds / 3600, 1e2 * pd.Series(mix.sig_delta).pct_change(), lw=2, label="Mix")
axs["bl"].plot(seconds / 3600, 1e2 * pd.Series(mix.bunch_length).pct_change(), lw=2, label="Mix")

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
fig.suptitle("Percent change from previous second")

plt.legend(title="Recompute Rates")
plt.tight_layout()
plt.show()
