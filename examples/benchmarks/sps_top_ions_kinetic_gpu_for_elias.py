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

from xibs.inputs import BeamParameters, OpticsParameters
from xibs.formulary import _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta
from xibs.kicks import KineticKickIBS

logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stdout,
    format="[%(asctime)s] [%(levelname)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
)
warnings.filterwarnings("ignore")  # scipy integration routines might warn

# ------------------- #

context = xo.ContextCupy()
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
n_part = int(1e3)
particles = xp.generate_matched_gaussian_bunch(
    num_particles=n_part,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    line=line,
)


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


nturns = 250  # number of turns to loop for
ibs_step = 50  # frequency at which to re-compute the growth rates / kick coefficients in [turns]
turns = np.arange(nturns, dtype=int)  # array of turns

# Initialize the dataclasses
my_tbt = Records.init_zeroes(nturns)
my_tbt.update_at_turn(0, particles, twiss)

# Re-initialize the IBS classes to be sure
beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
opticsparams = OpticsParameters.from_line(line)
IBS = KineticKickIBS(beamparams, opticsparams)

# We loop here now
for turn in range(1, nturns):
    # ----- Potentially re-compute the ellitest_parts integrals and IBS growth rates ----- #
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"Turn {turn:d}: re-computing growth rates and kick coefficients\n")
        # We compute from values at the previous turn
        IBS.compute_kick_coefficients(particles)
    else:
        print(f"===== Turn {turn:d} =====")

    # ----- Apply IBS Kick and Track Turn ----- #
    IBS.apply_ibs_kick(particles)
    line.track(particles, num_turns=1)

    # ----- Compute Emittances from Particles State for my tracked particles & update records----- #
    my_tbt.update_at_turn(turn, particles, twiss)



# ----- Plotting ----- #

fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(15, 8))

# Plot from my kicks and tracking
axs["epsx"].plot(turns, my_tbt.epsilon_x * 1e8, lw=1.5, label="Xibs")
axs["epsy"].plot(turns, my_tbt.epsilon_y * 1e8, lw=1.5, label="Xibs")
axs["sigd"].plot(turns, my_tbt.sigma_delta * 1e3, lw=1.5, label="Xibs")
axs["bl"].plot(turns, my_tbt.bunch_length * 1e2, lw=1.5, label="Xibs")

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
