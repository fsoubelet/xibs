"""
Tests for kinetic kicks applied on xpart.Particles on GPU context 
"""
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

import time

from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS

logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stdout,
    format="[%(asctime)s] [%(levelname)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
)
warnings.filterwarnings("ignore")  # scipy integration routines might warn

# We test the cupy context - cpu context will be about 9 times slower
context = xo.ContextCupy()
#context = xo.ContextCpu()

# Load example line - SPS ion injection
filepath = Path(__file__).parent.parent.parent / "tests" / "inputs" / "lines" / "sps_injection_ions.json"
line = xt.Line.from_json(filepath.absolute())
line.build_tracker(context)
line.optimize_for_tracking()

# Nominal SPS Pb injection values 
bunch_intensity = int(3.5e8)  # from the test config
sigma_z = 0.15  # from the test config
nemitt_x = 1.3e-6  # from the test config
nemitt_y = 0.9e-6  # from the test config

# Let's get our parameters
beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
opticsparams = OpticsParameters.from_line(line)

rf_voltage = 3.0e6  # 1.7MV from the test config
harmonic_number = 4653
cavity = "actcse.31632"
line[cavity].lag = 0  # 0 if below transition, 180 if above
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
    #engine="single-rf-harmonic",
)

# Tracking length
nturns = 100  # number of turns to loop for
ibs_step = 50  # frequency at which to re-compute the growth rates / kick coefficients in [turns]
turns = np.arange(nturns, dtype=int)  # array of turns

# Re-initialize the IBS classes to be sure
beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
opticsparams = OpticsParameters.from_line(line)
IBS = KineticKickIBS(beamparams, opticsparams)

# Track without storing any values
time00 = time.time()
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
    else:
        print(f"===== Turn {turn:d} =====")

    # ----- Apply IBS Kick and Track Turn ----- #
    IBS.apply_ibs_kick(particles)
    line.track(particles, num_turns=1)

# Check
time01 = time.time()
dt0 = time01-time00
print(f'Tracking finished in {dt0} seconds.')