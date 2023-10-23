"""
Utility functions (could be in cpymadtools at some point).
"""
from typing import Dict, Tuple

import numba
import numpy as np

from cpymad.madx import Madx
from scipy.constants import c


def get_madx_ibs_growth_rates(madx: Madx) -> Tuple[float, float, float]:
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
    """
    Takes values from loaded yaml config file and sets up the MAD-X lattice,
    sequence and beam so we can call IBS later on. This is machine / config
    agnostic and will run the same (working) logic for whatever is provided.
    """
    # ----- Get values from config (loaded from file) ----- #
    sequence = config["sequence"]  # sequence file location, relative to pytest root dir
    # opticsfile = config["opticsfile"]  # optics file location, relative to pytest root dir
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

    # ----- Beam, sequence and get parameters from Twiss & Summ tables ----- #
    madx.command.beam(particle=particle, energy=energy_GeV, mass=particle_mass_GeV, charge=particle_charge)
    madx.call(file=sequence)  # TODO: see above
    madx.use(sequence=sequence_name)
    madx.command.twiss()  # needed to have TWISS and SUMM tables
    circumference = madx.table.summ.length[0]  # ring circumference in [m]
    gamma_transition = madx.table.summ.gammatr[0]  # transition energy gamma
    gamma_rel = madx.table.twiss.summary.gamma  # relativistic gamma
    beta_rel = np.sqrt(1.0 - 1.0 / gamma_rel**2)  # relativistic beta
    etap = abs(1.0 / gamma_transition**2 - 1.0 / gamma_rel**2)  # get the slip factor
    beam_energy_GeV = madx.table.twiss.summary.energy  # beam energy in [GeV]
    bunch_length_m = config["blns"] * 1e-9 / 4.0 * (c * beta_rel)  # bunch length in [m]
    geom_epsx = norm_epsx / (gamma_rel * beta_rel)  # geom emit x in [m]
    geom_epsy = norm_epsy / (gamma_rel * beta_rel)  # geom emit y in [m]
    U0 = 0  # the energy loss per turn
    dpp = _bl_to_dpp(  # relative momentum spread from bunch length
        circumference,
        beam_energy_GeV,
        particle_charge,
        rf_voltage,
        U0,
        beta_rel,
        harmonic_number,
        etap,
        bunch_length_m,
    )
    dee = dpp * beta_rel**2  # relative energy spread [sigma_e in my codes]

    # ----- RF system ----- #
    # In MAD-X there is a convention that RF cavity if we activate it we need to multiply
    # the voltage by the particle charge (important for ions) -> MAD-X asks for q*V [now xsuite also does]
    madx.input(f"{rf_knobs}={rf_voltage*1e3}*{particle_charge};")
    madx.command.twiss()

    # ----- Re-use sequence, set beam properties and do final twiss ----- #
    madx.use(sequence=sequence_name)
    # VERY IMPORTANT: use madx.sequence[sequence_name].beam to set the beam properties as since cpymad v1.13.0
    # using madx.beam refers to the 'default_beam' which is not assigned to the sequence in question!
    madx.sequence[sequence_name].beam.ex = geom_epsx  # set the geom emit x (in [m])
    madx.sequence[sequence_name].beam.ey = geom_epsy  # set the geom emit y (in [m])
    madx.sequence[sequence_name].beam.sigt = bunch_length_m  # set the bunch length (in [m])
    madx.sequence[sequence_name].beam.sige = dee  # set the relative energy spread
    madx.sequence[sequence_name].beam.npart = bunch_intensity  # number of particles per bunch
    madx.command.twiss(centre=True)  # TODO: does using center in TWISS matter for the IBS calculations?


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
