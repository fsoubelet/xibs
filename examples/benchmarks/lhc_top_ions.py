"""
Benchmarking new code vs old code for analytical emittance evolutions
in the LHC, for ions at top energy.
"""
import time
import warnings

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import xpart as xp
import xtrack as xt

from xibs._old_michalis import MichalisIBS
from xibs.analytical import NagaitsevIBS
from xibs.formulary import ion_bunch_length
from xibs.inputs import BeamParameters, OpticsParameters

warnings.filterwarnings("ignore")  # scipy integration routines might warn
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 15,
        "figure.titlesize": 18,
    }
)

# ----- File and parameters (taken from lhc_top_ions.yml) ----- #
line_file = "../../tests/inputs/lines/lhc_top_ions.json"
harmonic_number = 34640
geom_epsx = 5.731420724345339e-10
geom_epsy = 5.731420724345339e-10
sig_delta = 0.00011981734527026258
bunch_length_m = 0.08993773197413962
rf_voltage = 14  # in MV
energy_loss = 0  # let's pretend
bunch_intensity = 1.8e8
nemitt_x = 1.65
nemitt_y = 1.65

# ----- Line and Twiss ----- #
line = xt.Line.from_json(line_file)
p0 = line.particle_ref
line.build_tracker()
twiss = line.twiss(method="4d")


# ----- Old and New APIs ----- #
beam_params = BeamParameters(p0)
beam_params.n_part = bunch_intensity
optics = OpticsParameters(twiss)
IBS = NagaitsevIBS(beam_params, optics)

MIBS = MichalisIBS()
MIBS.set_beam_parameters(p0)
MIBS.Npart = beam_params.n_part  # need to update this too
MIBS.set_optic_functions(twiss)


# ----- Dataclasses to store results ----- #
@dataclass
class Records:
    """Dataclass to store (and update) important values through tracking."""

    epsilon_x: np.ndarray
    epsilon_y: np.ndarray
    sig_delta: np.ndarray
    bunch_length: np.ndarray


nturns = 1000  # number of turns to loop for
ibs_step = 50  # frequency at which to re-compute the growth rates in [turns]
dt = 1 / IBS.optics.revolution_frequency  # this is the default anyway

# Structure to store results of the new codes
turn_by_turn = Records(
    epsilon_x=np.zeros(nturns, dtype=float),
    epsilon_y=np.zeros(nturns, dtype=float),
    sig_delta=np.zeros(nturns, dtype=float),
    bunch_length=np.zeros(nturns, dtype=float),
)

# Structure to store results of the old codes
old_turn_by_turn = Records(
    epsilon_x=np.zeros(nturns, dtype=float),
    epsilon_y=np.zeros(nturns, dtype=float),
    sig_delta=np.zeros(nturns, dtype=float),
    bunch_length=np.zeros(nturns, dtype=float),
)

# Store the initial values
turn_by_turn.bunch_length[0] = old_turn_by_turn.bunch_length[0] = bunch_length_m
turn_by_turn.sig_delta[0] = old_turn_by_turn.sig_delta[0] = sig_delta
turn_by_turn.epsilon_x[0] = old_turn_by_turn.epsilon_x[0] = geom_epsx
turn_by_turn.epsilon_y[0] = old_turn_by_turn.epsilon_y[0] = geom_epsy


# ----- Quick check for equality of growth rates from initial values above ----- #

IBS.integrals(turn_by_turn.epsilon_x[0], turn_by_turn.epsilon_y[0], turn_by_turn.sig_delta[0])
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
        IBS.integrals(
            turn_by_turn.epsilon_x[turn - 1],
            turn_by_turn.epsilon_y[turn - 1],
            turn_by_turn.sig_delta[turn - 1],
        )
        IBS.growth_rates(
            turn_by_turn.epsilon_x[turn - 1],
            turn_by_turn.epsilon_y[turn - 1],
            turn_by_turn.sig_delta[turn - 1],
            turn_by_turn.bunch_length[turn - 1],
        )

    # Compute the new emittances
    new_emit_x, new_emit_y, new_sig_delta = IBS.emittance_evolution(
        geom_epsx=turn_by_turn.epsilon_x[turn - 1],
        geom_epsy=turn_by_turn.epsilon_y[turn - 1],
        sigma_delta=turn_by_turn.sig_delta[turn - 1],
        # dt = 1.0 / IBS.optics.revolution_frequency,  # default value
    )

    # compute bunch length analytically as the Particles object hasn't changed
    sigma_e = new_sig_delta * IBS.beam_parameters.beta_rel**2
    bunch_l = ion_bunch_length(
        optics.circumference,
        harmonic_number,
        beam_params.total_energy_GeV,
        optics.slip_factor,
        sigma_e,
        beam_params.beta_rel,
        rf_voltage * 1e-3,  # has to be provided in [MV]
        beam_params.particle_charge,
    )

    # Update the records with the new values
    turn_by_turn.bunch_length[turn] = bunch_l
    turn_by_turn.sig_delta[turn] = new_sig_delta
    turn_by_turn.epsilon_x[turn] = new_emit_x
    turn_by_turn.epsilon_y[turn] = new_emit_y
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
    new_emit_x, new_emit_y, new_sig_delta = MIBS.emit_evol(
        Emit_x=old_turn_by_turn.epsilon_x[turn - 1],
        Emit_y=old_turn_by_turn.epsilon_y[turn - 1],
        Sig_M=old_turn_by_turn.sig_delta[turn - 1],
        BunchL=old_turn_by_turn.bunch_length[turn - 1],
        dt=dt,
    )

    # Compute bunch length analytically as the Particles object hasn't changed
    sigma_e = new_sig_delta * MIBS.betar**2
    bunch_l = ion_bunch_length(
        optics.circumference,
        harmonic_number,
        beam_params.total_energy_GeV,
        optics.slip_factor,
        sigma_e,
        beam_params.beta_rel,
        rf_voltage * 1e-3,  # has to be provided in [MV]
        beam_params.particle_charge,
    )

    # Update the records with the new values
    old_turn_by_turn.bunch_length[turn] = bunch_l
    old_turn_by_turn.sig_delta[turn] = new_sig_delta
    old_turn_by_turn.epsilon_x[turn] = new_emit_x
    old_turn_by_turn.epsilon_y[turn] = new_emit_y
end2 = time.time()

print(f"\nNew code took {end1 - start1:.3f} seconds")
print(f"Old code took {end2 - start2:.3f} seconds")
print(f"New code was ~{(end2 - start2)/(end1 - start1):.1f} faster")

# ----- Plot the results ----- #
figure, (epsx, epsy, sigdelta) = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

epsx.plot(old_turn_by_turn.epsilon_x, "o", ms=2, label="Old")
epsy.plot(old_turn_by_turn.epsilon_y, "o", ms=2, label="Old")
sigdelta.plot( old_turn_by_turn.sig_delta, "o", ms=2, label="Old")
epsx.plot(turn_by_turn.epsilon_x, label="New")
epsy.plot(turn_by_turn.epsilon_y, label="New")
sigdelta.plot(turn_by_turn.sig_delta, label="New")
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
