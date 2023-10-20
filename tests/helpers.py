"""
Utility functions (could be in cpymadtools at some point).
"""
import math

from typing import Dict, Tuple

import numba
import numpy as np

from cpymad.madx import Madx
from scipy.constants import c


def get_madx_ibs_beam_size_growth_time(madx: Madx) -> Tuple[float, float, float]:
    """
    Calls IBS module in MAD-X and return the horizontal, vertical and longitudinal growth rates.
    CAREFUL: the beam and twiss commands MUST have been called before calling this function.

    Args:
        madx (cpymad.madx.Madx): an instantiated `~cpymad.madx.Madx` object.

    Returns:
        A tuple with the values of the horizontal, vertical and longitudinal growth rates.
    """
    madx.command.ibs()
    madx.input("Tx=1/ibs.tx; Ty=1/ibs.ty; Tl=1/ibs.tl;")
    return madx.globals.Tx, madx.globals.Ty, madx.globals.Tl


def setup_madx_from_config(madx: Madx, config: Dict) -> None:
    """Takes values from loaded yaml config and sets up the MAD-X lattice, sequence and beam."""
    # Get values from the configuration (loaded from file)
    sequence = config["sequence"]  # TODO: relative paths here, needs fix or assume it's already called
    energy_GeV = config["energy"]  # beam energy in [GeV]
    norm_epsx = config["emit_x"] * 1e-6  # norm emit x in [m]
    norm_epsy = config["emit_y"] * 1e-6  # norm emit y in [m]
    bunch_intensity = config["bunch_intensity"]  # number of particles per bunch
    rf_knobs = config["cc_name_knobs"]  # MAD-X knobs for RF cavities
    rf_voltage = config["V0max"] * 1e-3  # RF voltage in [kV] (MV * 1e-3)
    harmonic_number = config["h"]  # RF harmonic number
    particle = config["particle"]  # particle type
    sequence_name = config["sequence_name"]  # accelerator sequence to use
    particle_mass_GeV = config["mass"]  # particle rest mass in [GeV]
    particle_classical_radius_m = config["radius"]  # classical particle radius
    particle_charge = config["charge"]  # particle charge in [e]

    # Beam, sequence and get some parameters from MAD-X Twiss
    madx.command.beam(particle=particle, energy=energy_GeV, mass=particle_mass_GeV, charge=particle_charge)
    madx.call(file=sequence)  # TODO: see above
    madx.use(sequence=sequence_name)
    madx.command.twiss()
    summ = madx.table.summ.dframe()
    circumference = summ["length"].to_numpy()[0]  # ring circumference
    gamma_transition = summ["gammatr"].to_numpy()[0]  # transition energy gamma
    gamma_rel = madx.table.twiss.summary.gamma  # relativistic gamma
    beta_rel = np.sqrt(1.0 - 1.0 / gamma_rel**2)  # relativistic beta
    etap = abs(1.0 / gamma_transition**2 - 1.0 / gamma_rel**2)  # get the slip factor
    beam_energy_GeV = madx.table.twiss.summary.energy  # beam energy in [GeV]

    # In MAD-X there is a convention that RF cavity if we activate it we need to multiply
    # the voltage by the particle charge (important for ions) -> MAD-X asks for q*V [now xsuite also does]
    madx.input(f"{rf_knobs}={rf_voltage*1e3}*{particle_charge};")
    U0 = 0  # the energy loss per turn
    bunch_length_m = config["blns"] * 1e-9 / 4.0 * (c * beta_rel)  # bunch length in [m]
    geom_epsx = norm_epsx / (gamma_rel * beta_rel)  # geom emit x in [m]
    geom_epsy = norm_epsy / (gamma_rel * beta_rel)  # geom emit y in [m]

    # fmt: off
    dpp = _bl_to_dpp(  # relative momentum spread from bunch length
        circumference, beam_energy_GeV, particle_charge, rf_voltage, U0, beta_rel, harmonic_number, etap, bunch_length_m
    )
    dee = dpp * beta_rel**2  # relative energy spread [sigma_e in my codes]
    # fmt: on

    # Twiss with RF system on, set some properties for the beam, and a final twiss
    madx.use(sequence=sequence_name)
    madx.command.twiss()
    madx.beam.ex = geom_epsx  # set the geom emit x (in [m])
    madx.beam.ey = geom_epsy  # set the geom emit y (in [m])
    madx.beam.sigt = bunch_length_m  # set the bunch length (in [m])
    madx.beam.sige = dee  # set the relative energy spread
    madx.beam.npart = bunch_intensity  # number of particles per bunch
    madx.command.twiss(centre=True)  # TODO: does using center in TWISS matter for the IBS calculations?
    # Getting the growth rates from MAD-X is left to the caller


# ----- Private functions ----- #


@numba.njit()
def _bl_to_dpp(
    RC: float,
    En: float,
    nc: int,
    RF_voltage: float,
    U0: float,
    betar: float,
    harm_number: int,
    etap: float,
    bl_i: float,
) -> float:
    """Get dpp from bunch length. Copied from old benchmark scripts of Sofia."""
    return (
        np.sqrt(2 / np.pi)
        * np.sqrt(nc * (RF_voltage - U0) * (np.sin(bl_i * harm_number * np.pi * betar / RC)) ** 2)
        / np.sqrt(En * harm_number * abs(etap))
        / betar
    )
