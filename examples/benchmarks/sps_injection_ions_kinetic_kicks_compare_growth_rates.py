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
filepath = Path(__file__).parent.parent.parent / "tests" / "inputs" / "lines" / "sps_injection_ions.json"
line = xt.Line.from_json(filepath.absolute())
line.build_tracker(context)
line.optimize_for_tracking()
twiss = line.twiss(method="4d")

# Test with normal values
bunch_intensity = int(3.5e8)  # from the test config
sigma_z = 0.22  # from the test config
nemitt_x = 1.2e-6  # from the test config
nemitt_y = 0.9e-6  # from the test config

# Let's get our parameters
beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
opticsparams = OpticsParameters.from_line(line)

rf_voltage = 3.0e6  # 1.7MV from the test config
harmonic_number = 4653
cavity = "actcse.31632"
line[cavity].lag = 0  # 0 if below transition, 180 if above - at injection we are below transition
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


# Set up a dataclass to store the results - also growth rates 
@dataclass
class Records:
    epsilon_x: np.ndarray
    epsilon_y: np.ndarray
    sigma_delta: np.ndarray
    bunch_length: np.ndarray
    Tx: np.ndarray
    Ty: np.ndarray
    Tz: np.ndarray

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
            Tx=np.zeros(n_turns, dtype=float),
            Ty=np.zeros(n_turns, dtype=float),
            Tz=np.zeros(n_turns, dtype=float)
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

# Calculate initial growth rates and initialize
kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
growth_rates = NIBS.growth_rates(analytical_tbt.epsilon_x[0], analytical_tbt.epsilon_y[0], 
                                 analytical_tbt.sigma_delta[0], analytical_tbt.bunch_length[0])
my_tbt.Tx[0] = kinetic_kick_coefficients.Kx
my_tbt.Ty[0] = kinetic_kick_coefficients.Ky
my_tbt.Tz[0] = kinetic_kick_coefficients.Kz
analytical_tbt.Tx[0] = growth_rates.Tx
analytical_tbt.Ty[0] = growth_rates.Ty
analytical_tbt.Tz[0] = growth_rates.Tz

print(kinetic_kick_coefficients)
print(growth_rates)

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
        kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
        MIBS.calculate_kinetic_coefficients(particles2)
        growth_rates = NIBS.growth_rates(  # recomputes integrals by default
            analytical_tbt.epsilon_x[turn - 1],
            analytical_tbt.epsilon_y[turn - 1],
            analytical_tbt.sigma_delta[turn - 1],
            analytical_tbt.bunch_length[turn - 1],
        )
        print(kinetic_kick_coefficients)
        print(growth_rates)
    else:
        print(f"===== Turn {turn:d} =====")

    my_tbt.Tx[turn] = kinetic_kick_coefficients.Kx
    my_tbt.Ty[turn] = kinetic_kick_coefficients.Ky
    my_tbt.Tz[turn] = kinetic_kick_coefficients.Kz
    analytical_tbt.Tx[turn] = growth_rates.Tx
    analytical_tbt.Ty[turn] = growth_rates.Ty
    analytical_tbt.Tz[turn] = growth_rates.Tz

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
fig.suptitle("SPS Injection Ions: Kinetic Kicks")
plt.tight_layout()


# Plot growth rates
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16,5))

ax1.plot(turns, analytical_tbt.Tx, alpha=0.7, lw=1.5, label='Analytical Nagaitsev')
ax1.plot(turns, my_tbt.Tx, label='Kinetic')

ax2.plot(turns, analytical_tbt.Ty, alpha=0.7, lw=1.5, label='Analytical Nagaitsev')
ax2.plot(turns, my_tbt.Ty, label='Kinetic')

ax3.plot(turns, analytical_tbt.Tz, alpha=0.7, lw=1.5, label='Analytical Nagaitsev')
ax3.plot(turns, my_tbt.Tz, label='Kinetic')

ax1.set_ylabel(r'$T_{x}$')
ax1.set_xlabel('Turns')

ax2.set_ylabel(r'$T_{y}$')
ax2.set_xlabel('Turns')

ax3.set_ylabel(r'$T_{z}$')
ax3.set_xlabel('Turns')
ax1.legend(fontsize=12)

plt.tight_layout()

plt.show()