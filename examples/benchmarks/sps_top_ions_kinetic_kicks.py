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
from xibs.formulary import _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta
from xibs.kicks import KineticKickIBS
from xibs._old_michalis import MichalisIBS

warnings.filterwarnings("ignore")

# ----- Load line and build tracker ----- #

context = xo.ContextCpu(omp_num_threads="auto")
xibs_repo = Path(__file__).absolute().parent.parent.parent
filepath = xibs_repo / "tests" / "inputs" / "lines" / "sps_top_ions.json"
line = xt.Line.from_json(filepath.absolute())
line.build_tracker(context)
line.optimize_for_tracking()
twiss = line.twiss(method="4d")

# ----- Activate RF Systems ----- #

rf_voltage = 1.7e6  # 1.7MV from the test config
harmonic_number = 4653
cavity = "actcse.31632"
line[cavity].lag = 180  # 0 if below transition, 180 if above
line[cavity].voltage = rf_voltage  # In Xsuite for ions, do not multiply by charge as in MADX
line[cavity].frequency = OpticsParameters.from_line(line).revolution_frequency * harmonic_number

# ----- Beam and Simulation Parameters ----- #

# Using fake values for beam parameters to be in a regime that 'stimulates' IBS
bunch_intensity = int(3.5e11)
sigma_z = 5e-2
nemitt_x = 1.0e-6
nemitt_y = 0.25e-6
n_part = int(1e3)
nturns = 1000  # number of turns to loop for
ibs_step = 50  # frequency at which to re-compute coefficients in [turns]

# ----- Create particles ----- #

particles = xp.generate_matched_gaussian_bunch(
    num_particles=n_part,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    particle_ref=line.particle_ref,
    line=line,
)
particles2 = particles.copy()

# ----- Compute initial (geometrical) emittances & bunch length all in [m] ----- #

sig_delta = _sigma_delta(particles)
bunch_l = _bunch_length(particles)
geom_epsx = _geom_epsx(particles, twiss.betx[0], twiss.dx[0])
geom_epsy = _geom_epsy(particles, twiss.bety[0], twiss.dy[0])

# ----- Dataclass to store results ----- #

@dataclass
class Records:
    epsilon_x: np.ndarray
    epsilon_y: np.ndarray
    sigma_delta: np.ndarray
    bunch_length: np.ndarray

    def update_at_turn(self, turn: int, parts: xp.Particles, twiss: xt.TwissTable):
        """Automatically update the records at given turn from the xtrack.Particles."""
        self.epsilon_x[turn] = _geom_epsx(parts, twiss.betx[0], twiss.dx[0])
        self.epsilon_y[turn] = _geom_epsy(parts, twiss.bety[0], twiss.dy[0])
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

# Initialize the dataclasses & store initial values
kicked_tbt = Records.init_zeroes(nturns)
old_tbt = Records.init_zeroes(nturns)
analytical_tbt = Records.init_zeroes(nturns)

kicked_tbt.update_at_turn(0, particles, twiss)
old_tbt.update_at_turn(0, particles, twiss)
analytical_tbt.update_at_turn(0, particles, twiss)

# ----- Initialize our IBS models (old and new) ----- #

beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
opticsparams = OpticsParameters.from_line(line)

IBS = KineticKickIBS(beamparams, opticsparams)
NIBS = NagaitsevIBS(beamparams, opticsparams)
MIBS = MichalisIBS()
MIBS.set_beam_parameters(particles)
MIBS.set_optic_functions(twiss)

# ----- We loop here now ----- # 

# We loop here now
for turn in range(1, nturns):
    # ----- Potentially re-compute the ellitest_parts integrals and IBS growth rates ----- #
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"Turn {turn:d}: re-computing diffusion and friction terms")
        # Compute kick coefficients from the particle distribution at this moment
        IBS.compute_kick_coefficients(particles)
        MIBS.calculate_kinetic_coefficients(particles2)
        NIBS.growth_rates(
            analytical_tbt.epsilon_x[turn - 1],
            analytical_tbt.epsilon_y[turn - 1],
            analytical_tbt.sigma_delta[turn - 1],
            analytical_tbt.bunch_length[turn - 1],
        )
    else:
        print(f"Turn {turn:d}")

    # ----- Apply IBS Kick and Track Turn ----- #
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


# ----- Plotting ----- #

turns = np.arange(nturns, dtype=int)  # array of turns
fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(15, 8))

# Plot from my kicks and tracking
axs["epsx"].plot(turns, kicked_tbt.epsilon_x * 1e8, lw=1.5, label="Xibs")
axs["epsy"].plot(turns, kicked_tbt.epsilon_y * 1e8, lw=1.5, label="Xibs")
axs["sigd"].plot(turns, kicked_tbt.sigma_delta * 1e3, lw=1.5, label="Xibs")
axs["bl"].plot(turns, kicked_tbt.bunch_length * 1e2, lw=1.5, label="Xibs")

# Plot from Michalis kicks and tracking
axs["epsx"].plot(turns, old_tbt.epsilon_x * 1e8, "-", lw=0.65, label="Michalis")
axs["epsy"].plot(turns, old_tbt.epsilon_y * 1e8, "-", lw=0.65, label="Michalis")
axs["sigd"].plot(turns, old_tbt.sigma_delta * 1e3, "-", lw=0.65, label="Michalis")
axs["bl"].plot(turns, old_tbt.bunch_length * 1e2, "-", lw=0.65, label="Michalis")

# Plot from analytical values
axs["epsx"].plot(turns, analytical_tbt.epsilon_x * 1e8, lw=1, label="Analytical")
axs["epsy"].plot(turns, analytical_tbt.epsilon_y * 1e8, lw=1, label="Analytical")
axs["sigd"].plot(turns, analytical_tbt.sigma_delta * 1e3, lw=1, label="Analytical")
axs["bl"].plot(turns, analytical_tbt.bunch_length * 1e2, lw=1, label="Analytical")

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
