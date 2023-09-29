"""
This is the old simulation script from Michalis, with no changes (other than formating
and using line to twiss and track, as using the tracker was deprecated).
"""
# IBS example: runs simple kick, kinetic theory and analytical
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xobjects as xo
import xpart as xp
import xtrack as xt

from cpymad.madx import Madx

# from lib.Xsuite_eval_emit_sig import *
# from lib.general_functions import *
from lib.IBSfunctions import *
from scipy.constants import e as qe
from scipy.constants import m_e, m_p

## Read a MAD-X Sequence
n_slice_per_element = 4
mad = Madx()
mad.call("inputs/chrom-corr_DR.newlattice_2GHz.seq")
mad.input(
    f"""
beam, particle=positron, energy=2.86, bunched;
use, sequence=RING;"""
)
twthick = mad.twiss().dframe()
mad.input(
    f"""
select, flag=MAKETHIN, SLICE={n_slice_per_element}, thick=false;
select, flag=MAKETHIN, pattern=wig, slice=1;
MAKETHIN, SEQUENCE=ring, MAKEDIPEDGE=true;
use, sequence=RING;"""
)
twthin = mad.twiss().dframe()
line = xt.Line.from_madx_sequence(mad.sequence["RING"])

line.remove_inactive_multipoles(inplace=True)
line.remove_zero_length_drifts(inplace=True)
line.merge_consecutive_drifts(inplace=True)
line.merge_consecutive_multipoles(inplace=True)

# Need to find if the aperture is changed in Xsuite!!!!!! #
# line.insert_element(idx = 0, # start of the ring
#                     element = xt.LimitRect(min_x = -0.04, max_x = 0.04, min_y = -0.04, max_y = 0.04),
#                     name = 'aperture')

## Choose a context
context = xo.ContextCpu()  # For CPU

## Transfer lattice on context and compile tracking code
tracker = xt.Tracker(_context=context, line=line, extra_headers=["#define XTRACK_MULTIPOLE_NO_SYNRAD"])

cavs = [ee for ee in line.elements if isinstance(ee, xt.Cavity)]
for cc in cavs:
    cc.lag = 180
    # if cc.voltage > 0:
    #    cc.voltage = -cc.voltage

p0 = xp.Particles(mass0=xp.ELECTRON_MASS_EV, q0=1, p0c=2.86e9)
tw = line.twiss(particle_ref=p0)

# ----- Set initial parameters -----
bunch_intensity = 4.4e9
sigma_z = 1.58e-3
# n_part = int(5e3)
n_part = int(5000)
nemitt_x = 5.6644e-07
nemitt_y = 3.7033e-09
n_turns = 1000
# mode = 'kinetic' # 'simple', 'analytical'
Harmonic_Num = 2852
Energy_loss = 0
RF_Voltage = 4.5
IBS_step = 50.0  # turns to recompute IBS kick
# ----------------------------------

particles0 = xp.generate_matched_gaussian_bunch(
    num_particles=n_part,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    particle_ref=p0,
    tracker=tracker,
)

for mode in ["kinetic", "simple", "analytical"]:
    print(f"Model: {mode}")

    particles = particles0.copy()

    # ----- Initialize IBS -----
    IBS = NagaitsevIBS()
    IBS.set_beam_parameters(particles)
    IBS.set_optic_functions(tw)

    dt = 1.0 / IBS.frev  # consecutive turns/frev, used only for analytical,

    # emit = Evaluate_Sigma_Emit(10,20,10,20,20,20)
    # emit.define_bins_width(particles, tw)

    ## Initialize dictionaries
    turn_by_turn = {}

    # record_coord = ['x', 'px', 'y', 'py', 'zeta', 'delta', 'state', 'particle_id']
    # for nn in record_coord:
    #    turn_by_turn[nn] = np.zeros((n_turns, n_part), dtype = getattr(particles, nn).dtype)

    record_emit = ["eps_x", "eps_y", "sig_delta", "bl"]
    for nn in record_emit:
        turn_by_turn[nn] = np.zeros((n_turns), dtype=float)

    # --- Initialize
    sig_x = np.std(particles.x[particles.state > 0])
    sig_y = np.std(particles.y[particles.state > 0])
    sig_delta = np.std(particles.delta[particles.state > 0])
    turn_by_turn["bl"][0] = np.std(particles.zeta[particles.state > 0])
    turn_by_turn["sig_delta"][0] = sig_delta
    turn_by_turn["eps_x"][0] = (sig_x**2 - (tw["dx"][0] * sig_delta) ** 2) / tw["betx"][0]
    turn_by_turn["eps_y"][0] = sig_y**2 / tw["bety"][0]

    for i in range(1, n_turns):
        # import time
        # start = time.time()

        print("Turn = ", i)
        print("N_part = ", len(particles.x[particles.state > 0]))

        #    for nn in record_coord:
        #        turn_by_turn[nn][i,:] = getattr(particles, nn)

        if mode != "analytical":
            sig_x = np.std(particles.x[particles.state > 0])
            sig_y = np.std(particles.y[particles.state > 0])
            sig_delta = np.std(particles.delta[particles.state > 0])
            turn_by_turn["bl"][i] = np.std(particles.zeta[particles.state > 0])
            turn_by_turn["sig_delta"][i] = sig_delta
            turn_by_turn["eps_x"][i] = (sig_x**2 - (tw["dx"][0] * sig_delta) ** 2) / tw["betx"][0]
            turn_by_turn["eps_y"][i] = sig_y**2 / tw["bety"][0]

        if mode == "analytical":
            if (i % IBS_step == 0) or (i == 1):
                IBS.calculate_integrals(
                    turn_by_turn["eps_x"][i - 1],
                    turn_by_turn["eps_y"][i - 1],
                    turn_by_turn["sig_delta"][i - 1],
                    turn_by_turn["bl"][i - 1],
                )
            Emit_x, Emit_y, Sig_M = IBS.emit_evol(
                turn_by_turn["eps_x"][i - 1],
                turn_by_turn["eps_y"][i - 1],
                turn_by_turn["sig_delta"][i - 1],
                turn_by_turn["bl"][i - 1],
                dt,
            )

            Sigma_E = Sig_M * IBS.betar**2
            BunchL = BunchLength(
                IBS.Circu,
                Harmonic_Num,
                IBS.EnTot,
                IBS.slip,
                Sigma_E,
                IBS.betar,
                RF_Voltage * 1e-3,
                Energy_loss,
                IBS.Ncharg,
            )

            turn_by_turn["bl"][i] = BunchL
            turn_by_turn["sig_delta"][i] = Sig_M
            turn_by_turn["eps_x"][i] = Emit_x
            turn_by_turn["eps_y"][i] = Emit_y

        elif mode == "kinetic":
            if (i % IBS_step == 0) or (i == 1):
                IBS.calculate_kinetic_coefficients(particles)
            IBS.apply_kinetic_kick(particles)
        elif mode == "simple":
            if (i % IBS_step == 0) or (i == 1):
                IBS.calculate_simple_kick(particles)
            IBS.apply_simple_kick(particles)

        line.track(particles)
        # end = time.time()
        # print(end - start)

    Emitt = []
    Emitt.append(turn_by_turn["eps_x"])
    Emitt.append(turn_by_turn["eps_y"])
    Emitt.append(turn_by_turn["sig_delta"])
    Emitt.append(turn_by_turn["bl"])

    df = pd.DataFrame(np.array(Emitt).T, columns=["eps_x", "eps_y", "sig_delta", "bl"])
    df.index.name = "Turn"
    df.to_parquet(f"outputs/old_script_new_xsuite_{mode}.parquet")
