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

from xibs._old_michalis import MichalisIBS
from xibs.analytical import NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS

logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stdout,
    format="[%(asctime)s] [%(levelname)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
)
warnings.filterwarnings("ignore")  # scipy integration routines might warn

# ----- Helpers ----- #


def _sigma_delta(parts: xp.Particles) -> float:
    return np.std(parts.delta[parts.state > 0])


def _bunch_length(parts: xp.Particles) -> float:
    return np.std(parts.zeta[parts.state > 0])


def _geom_epsx(parts: xp.Particles, twiss: xt.TwissTable) -> float:
    sig_x = np.std(parts.x[parts.state > 0])
    sig_delta = _sigma_delta(parts)
    return (sig_x**2 - (twiss["dx"][0] * sig_delta) ** 2) / twiss["betx"][0]


def _geom_epsy(parts: xp.Particles, twiss: xt.TwissTable) -> float:
    sig_y = np.std(parts.y[parts.state > 0])
    sig_delta = _sigma_delta(parts)
    return (sig_y**2 - (twiss["dy"][0] * sig_delta) ** 2) / twiss["bety"][0]


# ------------------- #

context = xo.ContextCpu(omp_num_threads="auto")
# context = xo.ContextCpu()
filepath = Path(__file__).parent.parent.parent / "tests" / "inputs" / "lines" / "sps_top_ions.json"
line = xt.Line.from_json(filepath.absolute())
line.build_tracker(context)
line.optimize_for_tracking()
twiss = line.twiss(method="4d")

# Using fake values for beam parameters to be in a regime that 'stimulates' IBS
bunch_intensity = int(3.5e11)  # from the test config
sigma_z = 5e-2  # from the test config
nemitt_x = 1.0e-6  # from the test config
nemitt_y = 0.25e-6  # from the test config

# Let's get our parameters
beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
opticsparams = OpticsParameters.from_line(line)

rf_voltage = 1.7e6  # 1.7MV from the test config
harmonic_number = 4653
cavity = "actcse.31632"
line[cavity].lag = 180  # 0 if below transition, 180 if above
line[cavity].voltage = rf_voltage  # In Xsuite for ions, do not multiply by charge as in MADX
line[cavity].frequency = opticsparams.revolution_frequency * harmonic_number

# Re-create particles with less elements as tracking takes a while
n_part = int(2e3)
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


# Set up a dataclass to store the results
@dataclass
class Records:
    epsilon_x: np.ndarray
    epsilon_y: np.ndarray
    sigma_delta: np.ndarray
    bunch_length: np.ndarray

    def update_at_turn(self, turn: int, parts: xp.Particles, twiss: xt.TwissTable):
        self.epsilon_x[turn] = _geom_epsx(parts, twiss)
        self.epsilon_y[turn] = _geom_epsy(parts, twiss)
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


nturns = 1_000  # number of turns to loop for
ibs_step = 50  # frequency at which to re-compute the growth rates / kick coefficients in [turns]
turns = np.arange(nturns, dtype=int)  # array of turns

# Initialize the dataclasses
my_tbt = Records.init_zeroes(nturns)
michalis_tbt = Records.init_zeroes(nturns)
analytical_tbt = Records.init_zeroes(nturns)

# Store the initial values
my_tbt.update_at_turn(0, particles, twiss)
michalis_tbt.update_at_turn(0, particles2, twiss)
analytical_tbt.update_at_turn(0, particles, twiss)

# Re-initialize the IBS classes to be sure
beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
opticsparams = OpticsParameters.from_line(line)
IBS = KineticKickIBS(beamparams, opticsparams)
NIBS = NagaitsevIBS(beamparams, opticsparams)
MIBS = MichalisIBS()
MIBS.set_beam_parameters(line.particle_ref)
MIBS.Npart = bunch_intensity
MIBS.set_optic_functions(twiss)


# We loop here now
for turn in range(1, nturns):
    # ----- Potentially re-compute the ellitest_parts integrals and IBS growth rates ----- #
    if (turn % ibs_step == 0) or (turn == 1):
        print(
            "=" * 60 + "\n",
            f"Turn {turn:d}: re-computing growth rates and kick coefficients\n",
            "=" * 60,
        )
        # We compute from values at the previous turn
        IBS.compute_kick_coefficients(particles)
        MIBS.calculate_kinetic_coefficients(particles2)
        NIBS.growth_rates(  # recomputes integrals by default
            analytical_tbt.epsilon_x[turn - 1],
            analytical_tbt.epsilon_y[turn - 1],
            analytical_tbt.sigma_delta[turn - 1],
            analytical_tbt.bunch_length[turn - 1],
        )
    else:
        print(f"===== Turn {turn:d} =====")

    # ----- Apply IBS Kick and Track Turn ----- #
    IBS.apply_ibs_kick(particles)
    MIBS.apply_kinetic_kick(particles2)
    line.track(particles, num_turns=1)
    line.track(particles2, num_turns=1)

    # ----- Compute Emittances from Particles State for my tracked particles & update records----- #
    my_tbt.update_at_turn(turn, particles, twiss)

    # ----- Compute Emittances from Particles State for Michalis' tracked particles & update records----- #
    michalis_tbt.update_at_turn(turn, particles2, twiss)

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

fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(15, 8))

# Plot from my kicks and tracking
axs["epsx"].plot(turns, my_tbt.epsilon_x * 1e8, lw=1.5, label="Xibs")
axs["epsy"].plot(turns, my_tbt.epsilon_y * 1e8, lw=1.5, label="Xibs")
axs["sigd"].plot(turns, my_tbt.sigma_delta * 1e3, lw=1.5, label="Xibs")
axs["bl"].plot(turns, my_tbt.bunch_length * 1e2, lw=1.5, label="Xibs")

# Plot from Michalis kicks and tracking
axs["epsx"].plot(turns, michalis_tbt.epsilon_x * 1e8, "-", lw=0.75, label="Michalis")
axs["epsy"].plot(turns, michalis_tbt.epsilon_y * 1e8, "-", lw=0.75, label="Michalis")
axs["sigd"].plot(turns, michalis_tbt.sigma_delta * 1e3, "-", lw=0.75, label="Michalis")
axs["bl"].plot(turns, michalis_tbt.bunch_length * 1e2, "-", lw=0.75, label="Michalis")

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
fig.suptitle("SPS Top Ions: Kinetic Kicks")

plt.tight_layout()
plt.show()
