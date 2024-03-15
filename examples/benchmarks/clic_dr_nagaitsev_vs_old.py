"""
Benchmarking new code vs old code for analytical emittance evolutions
in the CLIC DR (as in the analytical demo but comparing both)
"""
import time
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import xpart as xp
import xtrack as xt

from xibs._old_michalis import MichalisIBS
from xibs.analytical import NagaitsevIBS
from xibs.formulary import _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta
from xibs.inputs import BeamParameters, OpticsParameters

warnings.filterwarnings("ignore")

# ----- Load line and build tracker ----- #

xibs_repo = Path(__file__).parent.parent.parent
filepath = xibs_repo / "lines" / "chrom-corr_DR.newlattice_2GHz.json"
line = xt.Line.from_json(filepath)
line.build_tracker(extra_headers=["#define XTRACK_MULTIPOLE_NO_SYNRAD"])
p0 = xt.Particles(mass0=xp.ELECTRON_MASS_EV, q0=1, p0c=2.86e9)
line.particle_ref = p0
twiss = line.twiss()

# ----- Activate RF Systems ----- #

cavities = [element for element in line.elements if isinstance(element, xt.Cavity)]
for cavity in cavities:
    cavity.lag = 180

# ----- Beam and Simulation Parameters ----- #

bunch_intensity = 4.4e9
sigma_z = 1.58e-3
nemitt_x = 5.66e-7
nemitt_y = 3.7e-9
n_part = int(5e3)
nturns = 1000  # number of turns to loop for
ibs_step = 250  # frequency at which to re-compute the growth rates in [turns]

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

# ----- Compute initial (geometrical) emittances & bunch length all in [m] ----- #

sig_delta = _sigma_delta(particles)
bunch_l = _bunch_length(particles)
geom_epsx = _geom_epsx(particles, twiss.betx[0], twiss.dx[0])
geom_epsy = _geom_epsy(particles, twiss.bety[0], twiss.dy[0])

# ----- Dataclass to store results ----- #


@dataclass
class Records:
    """Dataclass to store (and update) important values through tracking."""

    epsilon_x: np.ndarray  # geometric horizontal emittance in [m]
    epsilon_y: np.ndarray  # geometric vertical emittance in [m]
    sig_delta: np.ndarray  # momentum spread
    bunch_length: np.ndarray  # bunch length in [m]

    @classmethod
    def init_zeroes(cls, n_turns: int) -> Self:  # noqa: F821
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


# Initialize the dataclasses & store initial values
turn_by_turn = Records.init_zeroes(nturns)
old_turn_by_turn = Records.init_zeroes(nturns)

turn_by_turn.update_at_turn(0, geom_epsx, geom_epsy, sig_delta, bunch_l)
old_turn_by_turn.update_at_turn(0, geom_epsx, geom_epsy, sig_delta, bunch_l)

# ----- Initialize our IBS models (old and new) ----- #

beam_params = BeamParameters(particles)
optics = OpticsParameters(twiss)
IBS = NagaitsevIBS(beam_params, optics)

MIBS = MichalisIBS()
MIBS.set_beam_parameters(particles)
MIBS.set_optic_functions(twiss)

# ----- Quick check for equality of growth rates from initial values above ----- #

IBS.growth_rates(
    turn_by_turn.epsilon_x[0],
    turn_by_turn.epsilon_y[0],
    turn_by_turn.sig_delta[0],
    turn_by_turn.bunch_length[0],
)
MIBS.calculate_integrals(
    Emit_x=old_turn_by_turn.epsilon_x[0],
    Emit_y=old_turn_by_turn.epsilon_y[0],
    Sig_M=old_turn_by_turn.sig_delta[0],
    BunchL=old_turn_by_turn.bunch_length[0],
)

assert np.isclose(MIBS.Ixx, IBS.ibs_growth_rates.Tx)
assert np.isclose(MIBS.Iyy, IBS.ibs_growth_rates.Ty)
assert np.isclose(MIBS.Ipp, IBS.ibs_growth_rates.Tz)
print("Initial comparison of growth rates: Success!")

# ---------------------------------------- #
# ----- LOOP OVER TURNS FOR NEW CODE ----- #
# ---------------------------------------- #

start1 = time.time()
for turn in range(1, nturns):
    # Potentially re-compute the Nagaitsev integrals and growth rates
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"New code - Turn {turn:>3}: re-computing the Nagaitsev integrals and growth rates")
        # We compute from values at the previous turn
        IBS.growth_rates(
            turn_by_turn.epsilon_x[turn - 1],
            turn_by_turn.epsilon_y[turn - 1],
            turn_by_turn.sig_delta[turn - 1],
            turn_by_turn.bunch_length[turn - 1],
        )

    # Compute the new emittances
    new_emit_x, new_emit_y, new_sig_delta, new_bunch_length = IBS.emittance_evolution(
        turn_by_turn.epsilon_x[turn - 1],
        turn_by_turn.epsilon_y[turn - 1],
        turn_by_turn.sig_delta[turn - 1],
        turn_by_turn.bunch_length[turn - 1],
        # dt = 1.0 / IBS.optics.revolution_frequency,  # default value
    )

    # Update the records with the new values
    turn_by_turn.update_at_turn(turn, new_emit_x, new_emit_y, new_sig_delta, new_bunch_length)
end1 = time.time()

# ---------------------------------------- #
# ----- LOOP OVER TURNS FOR OLD CODE ----- #
# ---------------------------------------- #

start2 = time.time()
for turn in range(1, nturns):
    # Potentially re-compute the Nagaitsev integrals and growth rates
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"Old code - Turn {turn:>3}: re-computing the Nagaitsev integrals and growth rates")
        MIBS.calculate_integrals(
            Emit_x=old_turn_by_turn.epsilon_x[turn - 1],
            Emit_y=old_turn_by_turn.epsilon_y[turn - 1],
            Sig_M=old_turn_by_turn.sig_delta[turn - 1],
            BunchL=old_turn_by_turn.bunch_length[turn - 1],
        )

    # Compute the new emittances
    dt = 1.0 / IBS.optics.revolution_frequency
    new_emit_x, new_emit_y, new_sig_delta = MIBS.emit_evol(
        Emit_x=old_turn_by_turn.epsilon_x[turn - 1],
        Emit_y=old_turn_by_turn.epsilon_y[turn - 1],
        Sig_M=old_turn_by_turn.sig_delta[turn - 1],
        BunchL=old_turn_by_turn.bunch_length[turn - 1],
        dt=dt,
    )
    # Compute bunch length analytically as the Particles object hasn't changed
    bunch_l = old_turn_by_turn.bunch_length[turn - 1] * np.exp(dt * float(0.5 * MIBS.Ipp))

    # Update the records with the new values
    old_turn_by_turn.bunch_length[turn] = bunch_l
    old_turn_by_turn.sig_delta[turn] = new_sig_delta
    old_turn_by_turn.epsilon_x[turn] = new_emit_x
    old_turn_by_turn.epsilon_y[turn] = new_emit_y
end2 = time.time()

# ----- Print timing information ----- #

print(f"\nNew code took {end1 - start1:.3f} seconds")
print(f"Old code took {end2 - start2:.3f} seconds")
print(f"New code was ~{(end2 - start2)/(end1 - start1):.1f} faster")

# ----- Plot the results ----- #

figure, (epsx, epsy, sigdelta) = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

epsx.plot(1e10 * old_turn_by_turn.epsilon_x, "o", ms=2, label="Old")
epsy.plot(1e13 * old_turn_by_turn.epsilon_y, "o", ms=2, label="Old")
sigdelta.plot(1e3 * old_turn_by_turn.sig_delta, "o", ms=2, label="Old")
epsx.plot(1e10 * turn_by_turn.epsilon_x, label="New")
epsy.plot(1e13 * turn_by_turn.epsilon_y, label="New")
sigdelta.plot(1e3 * turn_by_turn.sig_delta, label="New")
epsx.legend()
epsy.legend()
sigdelta.legend()
epsx.set_ylabel(r"$\varepsilon_x$ [$10^{-10}$m]")
epsy.set_ylabel(r"$\varepsilon_y$ [$10^{-13}$m]")
sigdelta.set_ylabel(r"$\sigma_{\delta}$ [$10^{-3}$]")
sigdelta.set_xlabel("Turn Number")
figure.align_ylabels([epsx, epsy, sigdelta])

plt.tight_layout()
plt.show()
