"""
Benchmarking new code vs old code for analytical emittance evolutions
in the LHC, for protons at injection energy.
"""
import time
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from xibs._old_michalis import MichalisIBS
from xibs.analytical import NagaitsevIBS, BjorkenMtingwaIBS
from xibs.inputs import BeamParameters, OpticsParameters

warnings.filterwarnings("ignore")  # scipy integration routines might warn

# ----- Load line and build tracker ----- #

xibs_repo = Path(__file__).parent.parent.parent
filepath = xibs_repo / "tests" / "inputs" / "lines" / "lhc_injection_protons.json"
line = xt.Line.from_json(filepath)
p0 = line.particle_ref
line.build_tracker()
twiss = line.twiss(method="4d")

# ----- Beam and Simulation Parameters ----- #

geom_epsx = 3.75e-9
geom_epsy = 3.75e-9
sig_delta = 0.00024748579759502934
bunch_length_m = 0.08993754190079283
bunch_intensity = 1.8e11
n_turns = 1000  # number of turns to loop for
ibs_step = 50  # frequency at which to re-compute the growth rates in [turns]

# ----- Dataclasses to store results ----- #

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
nag_turn_by_turn = Records.init_zeroes(n_turns)
bm_turn_by_turn = Records.init_zeroes(n_turns)
old_turn_by_turn = Records.init_zeroes(n_turns)

nag_turn_by_turn.update_at_turn(0, geom_epsx, geom_epsy, sig_delta, bunch_length_m)
bm_turn_by_turn.update_at_turn(0, geom_epsx, geom_epsy, sig_delta, bunch_length_m)
old_turn_by_turn.update_at_turn(0, geom_epsx, geom_epsy, sig_delta, bunch_length_m)

# ----- Initialize our IBS models (old and new) ----- #

beam_params = BeamParameters.from_line(line, n_part=bunch_intensity)
optics = OpticsParameters.from_line(line)

NIBS = NagaitsevIBS(beam_params, optics)
BMIBS = BjorkenMtingwaIBS(beam_params, optics)
MIBS = MichalisIBS()
MIBS.set_beam_parameters(p0)
MIBS.Npart = beam_params.n_part  # need to update this too
MIBS.set_optic_functions(twiss)

# ----- Quick check for equality of growth rates from initial values above ----- #

NIBS.growth_rates(
    nag_turn_by_turn.epsilon_x[0],
    nag_turn_by_turn.epsilon_y[0],
    nag_turn_by_turn.sig_delta[0],
    nag_turn_by_turn.bunch_length[0],
)
BMIBS.growth_rates(
    bm_turn_by_turn.epsilon_x[0],
    bm_turn_by_turn.epsilon_y[0],
    bm_turn_by_turn.sig_delta[0],
    bm_turn_by_turn.bunch_length[0],
)
MIBS.calculate_integrals(
    Emit_x=old_turn_by_turn.epsilon_x[0],
    Emit_y=old_turn_by_turn.epsilon_y[0],
    Sig_M=old_turn_by_turn.sig_delta[0],
    BunchL=old_turn_by_turn.bunch_length[0],
)

assert np.isclose(MIBS.Ixx, NIBS.ibs_growth_rates.Tx, rtol=1e-2)
assert np.isclose(MIBS.Iyy, NIBS.ibs_growth_rates.Ty, rtol=1e-2)
assert np.isclose(MIBS.Ipp, NIBS.ibs_growth_rates.Tz, rtol=1e-2)
assert np.isclose(MIBS.Ixx, BMIBS.ibs_growth_rates.Tx, rtol=1e-2)
assert np.isclose(MIBS.Iyy, BMIBS.ibs_growth_rates.Ty, rtol=1e-2)
assert np.isclose(MIBS.Ipp, BMIBS.ibs_growth_rates.Tz, rtol=1e-2)
print("Initial comparison of growth rates: Success!")

# ---------------------------------------------- #
# ----- LOOP OVER TURNS FOR NAGAITSEV CODE ----- #
# ---------------------------------------------- #

start1 = time.time()
for turn in range(1, n_turns):
    # Potentially re-compute the Nagaitsev integrals and growth rates
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"New code - Turn {turn:>3}: re-computing the Nagaitsev integrals and growth rates")
        # We compute from values at the previous turn
        NIBS.growth_rates(
            nag_turn_by_turn.epsilon_x[turn - 1],
            nag_turn_by_turn.epsilon_y[turn - 1],
            nag_turn_by_turn.sig_delta[turn - 1],
            nag_turn_by_turn.bunch_length[turn - 1],
        )

    # Compute the new emittances
    new_emit_x, new_emit_y, new_sig_delta, new_bunch_length = NIBS.emittance_evolution(
        nag_turn_by_turn.epsilon_x[turn - 1],
        nag_turn_by_turn.epsilon_y[turn - 1],
        nag_turn_by_turn.sig_delta[turn - 1],
        nag_turn_by_turn.bunch_length[turn - 1],
        # dt = 1.0 / IBS.optics.revolution_frequency,  # default value
    )

    # Update the records with the new values
    nag_turn_by_turn.update_at_turn(turn, new_emit_x, new_emit_y, new_sig_delta, new_bunch_length)
end1 = time.time()


# ---------------------------------------------------- #
# ----- LOOP OVER TURNS FOR BJORKEN-MTINGWA CODE ----- #
# ---------------------------------------------------- #

start2 = time.time()
for turn in range(1, n_turns):
    # Potentially re-compute the Nagaitsev integrals and growth rates
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"New code - Turn {turn:>3}: re-computing the Nagaitsev integrals and growth rates")
        # We compute from values at the previous turn
        BMIBS.growth_rates(
            bm_turn_by_turn.epsilon_x[turn - 1],
            bm_turn_by_turn.epsilon_y[turn - 1],
            bm_turn_by_turn.sig_delta[turn - 1],
            bm_turn_by_turn.bunch_length[turn - 1],
        )

    # Compute the new emittances
    new_emit_x, new_emit_y, new_sig_delta, new_bunch_length = BMIBS.emittance_evolution(
        bm_turn_by_turn.epsilon_x[turn - 1],
        bm_turn_by_turn.epsilon_y[turn - 1],
        bm_turn_by_turn.sig_delta[turn - 1],
        bm_turn_by_turn.bunch_length[turn - 1],
        # dt = 1.0 / IBS.optics.revolution_frequency,  # default value
    )

    # Update the records with the new values
    bm_turn_by_turn.update_at_turn(turn, new_emit_x, new_emit_y, new_sig_delta, new_bunch_length)
end2 = time.time()

# ---------------------------------------- #
# ----- LOOP OVER TURNS FOR OLD CODE ----- #
# ---------------------------------------- #

start3 = time.time()
for turn in range(1, n_turns):
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
    dt = 1.0 / NIBS.optics.revolution_frequency
    new_emit_x, new_emit_y, new_sig_delta = MIBS.emit_evol(
        Emit_x=old_turn_by_turn.epsilon_x[turn - 1],
        Emit_y=old_turn_by_turn.epsilon_y[turn - 1],
        Sig_M=old_turn_by_turn.sig_delta[turn - 1],
        BunchL=old_turn_by_turn.bunch_length[turn - 1],
        dt=dt,
    )

    # Compute bunch length analytically as the Particles object hasn't changed
    new_bunch_l = old_turn_by_turn.bunch_length[turn - 1] * np.exp(dt * float(0.5 * MIBS.Ipp))

    # Update the records with the new values
    old_turn_by_turn.update_at_turn(turn, new_emit_x, new_emit_y, new_sig_delta, new_bunch_l)
end3 = time.time()

print(f"\nNew Nagaitsev code took {end1 - start1:.3f} seconds")
print(f"\nNew Bjorken-Mtingwa code took {end2 - start2:.3f} seconds")
print(f"Old code took {end3 - start3:.3f} seconds")
print(f"New Nagaitsev code was ~{(end3 - start3)/(end1 - start1):.1f} faster")
print(f"New Bjorken-Mtingwa code was ~{(end3 - start3)/(end2 - start2):.1f} faster")

# ----- Plot the results ----- #

figure, (epsx, epsy, sigdelta) = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

epsx.plot(old_turn_by_turn.epsilon_x, "o", ms=2, label="Old")
epsy.plot(old_turn_by_turn.epsilon_y, "o", ms=2, label="Old")
sigdelta.plot(old_turn_by_turn.sig_delta, "o", ms=2, label="Old")
epsx.plot(nag_turn_by_turn.epsilon_x, label="New Nagaitsev")
epsy.plot(nag_turn_by_turn.epsilon_y, label="New Nagaitsev")
sigdelta.plot(nag_turn_by_turn.sig_delta, label="New Nagaitsev")
epsx.plot(bm_turn_by_turn.epsilon_x, ls=":", label="New Bjorken-Mtingwa")
epsy.plot(bm_turn_by_turn.epsilon_y, ls=":", label="New Bjorken-Mtingwa")
sigdelta.plot(bm_turn_by_turn.sig_delta, ls=":", label="New Bjorken-Mtingwa")
epsx.legend()
epsy.legend()
sigdelta.legend()
epsx.set_ylabel(r"$\varepsilon_x$ [m]")
epsy.set_ylabel(r"$\varepsilon_y$ [m]")
sigdelta.set_ylabel(r"$\sigma_{\delta}$ [-]")
sigdelta.set_xlabel("Turn Number")
figure.align_ylabels([epsx, epsy, sigdelta])

plt.tight_layout()
plt.show()
