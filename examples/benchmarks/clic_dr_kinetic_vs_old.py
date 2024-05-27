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
filepath = xibs_repo / "examples" / "lines" / "chrom-corr_DR.newlattice_2GHz.json"
line = xt.Line.from_json(filepath)
line.build_tracker(context, extra_headers=["#define XTRACK_MULTIPOLE_NO_SYNRAD"])
p0 = xt.Particles(mass0=xp.ELECTRON_MASS_EV, q0=1, p0c=2.86e9)
line.particle_ref = p0
twiss = line.twiss()

# ----- Activate RF Systems ----- #

cavities = [element for element in line.elements if isinstance(element, xt.Cavity)]
for cavity in cavities:
    cavity.lag = 180

# ----- Beam and Simulation Parameters ----- #
    
bunch_intensity = int(4.5e9)
sigma_z = 1.58e-3
nemitt_x = 5.66e-7
nemitt_y = 3.7e-9
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
    particle_ref=p0,
    line=line,
)
particles2 = particles.copy()

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

for turn in range(1, nturns):
    # ----- Potentially re-compute the IBS growth rates and kick coefficients ----- #
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"Turn {turn:d}: re-computing diffusion and friction terms")
        # Compute kick coefficients from the particle distribution at this moment
        IBS.compute_kick_coefficients(particles)
        MIBS.calculate_kinetic_coefficients(particles2)
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

# ----- Plot the results ----- #

turns = np.arange(nturns, dtype=int)  # array of turns
fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(15, 7.5))

# Plot from tracked & kicked particles
l1, = axs["epsx"].plot(turns, kicked_tbt.epsilon_x * 1e10, lw=2, label="Xibs")
axs["epsy"].plot(turns, kicked_tbt.epsilon_y * 1e13, lw=2, label="Xibs")
axs["sigd"].plot(turns, kicked_tbt.sigma_delta * 1e3, lw=2, label="Xibs")
axs["bl"].plot(turns, kicked_tbt.bunch_length * 1e3, lw=2, label="Xibs")

# Plot from michalis results
l2, = axs["epsx"].plot(turns, old_tbt.epsilon_x * 1e10, lw=2, label="Michalis")
axs["epsy"].plot(turns, old_tbt.epsilon_y * 1e13, lw=2, label="Michalis")
axs["sigd"].plot(turns, old_tbt.sigma_delta * 1e3, lw=2, label="Michalis")
axs["bl"].plot(turns, old_tbt.bunch_length * 1e3, lw=2, label="Michalis")

# Plot from analytical values
l3, = axs["epsx"].plot(turns, analytical_tbt.epsilon_x * 1e10, lw=2.5, label="Nagaitsev")
axs["epsy"].plot(turns, analytical_tbt.epsilon_y * 1e13, lw=2.5, label="Nagaitsev")
axs["sigd"].plot(turns, analytical_tbt.sigma_delta * 1e3, lw=2.5, label="Nagaitsev")
axs["bl"].plot(turns, analytical_tbt.bunch_length * 1e3, lw=2.5, label="Nagaitsev")

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
    # axis.legend(loc=9, ncols=4)

fig.align_ylabels((axs["epsx"], axs["sigd"]))
fig.align_ylabels((axs["epsy"], axs["bl"]))

fig.suptitle("CLIC DR - Kinetic Kicks")
fig.legend(handles=(l1, l2, l3), loc="lower center", bbox_to_anchor=(0.5, 0.873), ncol=3, fontsize=14)

plt.tight_layout()
plt.show()
