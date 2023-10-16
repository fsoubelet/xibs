"""
Utility functions (could be in cpymadtools at some point).
"""
import math

from typing import Dict, Tuple

import numpy as np

from cpymad.madx import Madx
from scipy.constants import c


def re_cycle_sequence(madx: Madx, sequence: str = "sps", start: str = "sps$start") -> None:
    """
    Re-cycles the provided *sequence* from a different starting point, given as *start*.

    Args:
        madx (cpymad.madx.Madx): an instantiated `~cpymad.madx.Madx` object.
        sequence (str): the sequence to re-cycle.
        start (str): element to start the new cycle from.

    Example:
        .. code-block:: python

            >>> re_cycle_sequence(madx, sequence="sps", start="mbb.42070")
    """
    madx.command.seqedit(sequence=sequence)
    madx.command.flatten()
    madx.command.cycle(start=start)
    madx.command.endedit()


def make_sps_thin(madx: Madx, sequence: str, slicefactor: int = 1, **kwargs) -> None:
    """
    Executes the ``MAKETHIN`` command for the SPS sequence as previously done in ``MAD-X`` macros.
    This will by default use the ``teapot`` style and will enforce ``makedipedge``.

    One can find an exemple use of this function in the :ref:`AC Dipole Tracking <demo-ac-dipole-tracking>`
    and :ref:`Free Tracking <demo-free-tracking>` example galleries.

    Args:
        madx (cpymad.madx.Madx): an instantiated `~cpymad.madx.Madx` object.
        sequence (str): the sequence to use for the ``MAKETHIN`` command.
        slicefactor (int): the slice factor to apply in ``MAKETHIN``, which is a factor
            applied to default values for different elements, as did the old macro. Defaults
            to 1.
        **kwargs: any keyword argument will be transmitted to the ``MAD-X`` ``MAKETHN``
            command, namely ``style`` (will default to ``teapot``) and the ``makedipedge``
            flag (will default to `True`).
    """
    style = kwargs.get("style", "teapot")
    makedipedge = kwargs.get("makedipedge", False)  # defaults to False to compensate default TEAPOT style

    madx.command.use(sequence=sequence)
    madx.select(flag="makethin", clear=True)
    madx.select(flag="makethin", slice=slicefactor, thick=False)
    madx.command.makethin(sequence=sequence, style=style, makedipedge=makedipedge)


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
    # Define beam parameters
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

    # Sequence and beam
    madx.command.beam(particle=particle, energy=energy_GeV, mass=particle_mass_GeV, charge=particle_charge)
    madx.call(file=sequence)  # TODO: see above
    madx.use(sequence=sequence_name)
    madx.command.twiss()

    # Get some parameters from MAD-X Twiss
    twiss = madx.table.twiss.dframe()
    summ = madx.table.summ.dframe()
    circumference = summ["length"].to_numpy()[0]  # ring circumference
    gamma_transition = summ["gammatr"].to_numpy()[0]  # transition energy gamma
    gamma_rel = madx.table.twiss.summary.gamma  # relativistic gamma
    beta_rel = np.sqrt(1.0 - 1.0 / gamma_rel**2)  # relativistic beta
    etap = abs(1.0 / gamma_transition**2 - 1.0 / gamma_rel**2)  # ???
    beam_energy_GeV = madx.table.twiss.summary.energy  # beam energy in [GeV]
    # E0 = beam_energy_GeV / gamma_rel  # rest mass energy in [GeV] | this is particle_mass_GeV

    madx.input(f"{rf_knobs}={rf_voltage*1e3}*{particle_charge};")
    U0 = 0
    bunch_length_m = config["blns"] * 1e-9 / 4.0 * (c * beta_rel)  # bunch length in [m]
    # bunch_length_level = config["bl_lev"] * 1e-9 / 4.0 * (c * beta_rel)  # ???
    # bl_i = bunch_length_m
    geom_epsx = norm_epsx / (gamma_rel * beta_rel)  # geom emit x in [m]
    geom_epsy = norm_epsy / (gamma_rel * beta_rel)  # geom emit y in [m]

    revolution_frequency = beta_rel * c / circumference  # revolution frequency in [Hz]
    rf_cavity_frequency = harmonic_number * revolution_frequency  # RF cavity frequency in [Hz]
    rf_period = 1.0 / rf_cavity_frequency  # RF period in [s]

    dpp = _bl_to_dpp(circumference, beam_energy_GeV, particle_charge, rf_voltage, U0, beta_rel, harmonic_number, etap, bunch_length_m)  # relative momentum spread
    dee = dpp * beta_rel**2  # relative energy spread

    # Twiss sequence, this time with RF system on
    madx.use(sequence=sequence_name)
    madx.command.twiss()
    # twiss = madx.table.twiss.dframe()

    # Set some properties for the beam
    madx.beam.ex = geom_epsx  # set the geom emit x (in [m])
    madx.beam.ey = geom_epsy  # set the geom emit y (in [m])
    madx.beam.sigt = bunch_length_m  # set the bunch length (in [m])
    madx.beam.sige = dee  # set the relative energy spread
    madx.beam.npart = bunch_intensity  # number of particles per bunch

    # TODO: why are we setting these to 0?
    # U0, taux, tauy, taul, ex0, ey0, sp0, ss0 = (0,) * 8
    madx.command.twiss(centre=True)

    # TODO: is this necessary?
    # madx.input(
    #     f"""
    # init_ex = beam->Ex;
    # init_ey = beam->Ey;
    # bl_ns= beam->sigt/(clight*{betar})*4*1e9;
    # """
    # )

    # Now we are done, let's get the growth rates from MAD-X
    # but this is left to the caller


# ----- Private functions ----- #


def _dpp_to_bl(RC, En, nc, RF_voltage, U0, betar, harm_number, etap, dpp):
    """Get bunch length from dpp. Copied from old benchmark scripts of Sofia."""
    # fmt: off
    return (
        c * RC
        * math.acos(
            (En * nc * (RF_voltage - U0) * betar**2 - dpp**2 * En**2 * harm_number * np.pi * betar**4 * abs(etap))
            / (En * nc * (RF_voltage - U0) * betar**2)
        )
        / (2 * c * harm_number * np.pi * betar)
    )
    # fmt: on


def _bl_to_dpp(RC, En, nc, RF_voltage, U0, betar, harm_number, etap, bl_i):
    """Get dpp from bunch length. Copied from old benchmark scripts of Sofia."""
    return (
        np.sqrt(2 / np.pi)
        * np.sqrt(nc * (RF_voltage - U0) * (np.sin(bl_i * harm_number * np.pi * betar / RC)) ** 2)
        / np.sqrt(En * harm_number * abs(etap))
        / betar
    )
