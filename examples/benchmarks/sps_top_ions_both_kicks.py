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
from xibs.kicks import KineticKickIBS, SimpleKickIBS

logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stdout,
    format="[%(asctime)s] [%(levelname)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
)
warnings.simplefilter("ignore")  # for this tutorial's clarity

# ----- Helpers ----- #


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


# ------------------- #

context = xo.ContextCpu(omp_num_threads="auto")
filepath = Path(__file__).parent.parent.parent / "tests" / "inputs" / "lines" / "sps_top_ions.json"
line = xt.Line.from_json(filepath.absolute())
line.build_tracker(context)
line.optimize_for_tracking()
twiss = line.twiss(method="4d")

# Using fake values for beam parameters to be in a regime that 'stimulates' IBS
bunch_intensity = int(5e11)
sigma_z = 5e-2
nemitt_x = 1.0e-6
nemitt_y = 0.25e-6

rf_voltage = 1.7e6  # 1.7MV from the test config
harmonic_number = 4653
cavity = "actcse.31632"
line[cavity].lag = 180  # 0 if below transition, 180 if above
line[cavity].voltage = rf_voltage  # In Xsuite for ions, do not multiply by charge as in MADX
line[cavity].frequency = OpticsParameters.from_line(line).revolution_frequency * harmonic_number

# ------------------- #

beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
opticsparams = OpticsParameters.from_line(line)
KIBS = KineticKickIBS(beamparams, opticsparams)
SIBS = SimpleKickIBS(beamparams, opticsparams)
NIBS = NagaitsevIBS(beamparams, opticsparams)

n_part = int(2e3)  # 2000 particles should be enough for this example
particles = xp.generate_matched_gaussian_bunch(
    num_particles=n_part,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    line=line,
    engine="single-rf-harmonic",
)
particles2 = particles.copy()

# ------------------- #

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

nturns = 1_500  # number of turns to loop for
ibs_step = 50  # frequency at which to re-compute coefficients in [turns]

# Initialize the dataclasses
kinetic_tbt = Records.init_zeroes(nturns)
simple_tbt = Records.init_zeroes(nturns)
analytical_tbt = Records.init_zeroes(nturns)

# Store the initial values
kinetic_tbt.update_at_turn(0, particles, twiss)
simple_tbt.update_at_turn(0, particles, twiss)
analytical_tbt.update_at_turn(0, particles, twiss)


# ----- Tracking ----- #

for turn in range(1, nturns):
    # ----- Potentially re-compute the IBS growth rates and kick coefficients ----- #
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"Turn {turn:d}: re-computing diffusion and friction terms")
        # Compute kick coefficients from the particle distribution at this moment
        KIBS.compute_kick_coefficients(particles)
        SIBS.compute_kick_coefficients(particles2)
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
    KIBS.apply_ibs_kick(particles)
    SIBS.apply_ibs_kick(particles2)
    line.track(particles, num_turns=1)
    line.track(particles2, num_turns=1)

    # ----- Update records for tracked particles ----- #
    kinetic_tbt.update_at_turn(turn, particles, twiss)
    simple_tbt.update_at_turn(turn, particles2, twiss)

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

# ----- Plotting ----- #

turns = np.arange(nturns, dtype=int)  # array of turns
fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(15, 7.5))

# Plot from my kicks and tracking
axs["epsx"].plot(turns, kinetic_tbt.epsilon_x * 1e8, lw=1.5, label="Kinetic")
axs["epsy"].plot(turns, kinetic_tbt.epsilon_y * 1e8, lw=1.5, label="Kinetic")
axs["sigd"].plot(turns, kinetic_tbt.sigma_delta * 1e3, lw=1.5, label="Kinetic")
axs["bl"].plot(turns, kinetic_tbt.bunch_length * 1e2, lw=1.5, label="Kinetic")

# Plot from Michalis kicks and tracking
axs["epsx"].plot(turns, simple_tbt.epsilon_x * 1e8, "-", lw=0.75, label="Simple")
axs["epsy"].plot(turns, simple_tbt.epsilon_y * 1e8, "-", lw=0.75, label="Simple")
axs["sigd"].plot(turns, simple_tbt.sigma_delta * 1e3, "-", lw=0.75, label="Simple")
axs["bl"].plot(turns, simple_tbt.bunch_length * 1e2, "-", lw=0.75, label="Simple")

# Plot from analytical values
axs["epsx"].plot(turns, analytical_tbt.epsilon_x * 1e8, lw=1.5, label="Nagaitsev")
axs["epsy"].plot(turns, analytical_tbt.epsilon_y * 1e8, lw=1.5, label="Nagaitsev")
axs["sigd"].plot(turns, analytical_tbt.sigma_delta * 1e3, lw=1.5, label="Nagaitsev")
axs["bl"].plot(turns, analytical_tbt.bunch_length * 1e2, lw=1.5, label="Nagaitsev")

# Axes parameters
axs["epsx"].set_ylabel(r"$\varepsilon_x$ [$10^{-8}$m]")
axs["epsy"].set_ylabel(r"$\varepsilon_y$ [$10^{-8}$m]")
axs["sigd"].set_ylabel(r"$\sigma_{\delta}$ [$10^{-3}$]")
axs["bl"].set_ylabel(r"Bunch length [cm]")

for axis in (axs["epsy"], axs["bl"]):
    axis.yaxis.set_label_position("right")
    axis.yaxis.tick_right()

for axis in (axs["sigd"], axs["bl"]):
    axis.set_xlabel("Turn Number")

for axis in axs.values():
    axis.yaxis.set_major_locator(plt.MaxNLocator(4))
    axis.legend(loc=9, ncols=4)

fig.align_ylabels((axs["epsx"], axs["sigd"]))
fig.align_ylabels((axs["epsy"], axs["bl"]))
fig.suptitle("SPS Top Ions: Both Kick Formalism")

plt.tight_layout()
plt.show()
