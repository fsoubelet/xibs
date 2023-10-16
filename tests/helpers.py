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


def get_madx_setup_from_config(madx: Madx, config: Dict) -> None:
    """Takes values from loaded yaml config and sets up the MAD-X lattice, sequence and beam."""
    # Define beam parameters
    sequence = config["sequence"]  # TODO: relative paths here, needs fix or assume it's already called
    energy = config["energy"]
    exn = config["emit_x"] * 1e-6  # norm emit x in [m]
    eyn = config["emit_y"] * 1e-6  # norm emit y in [m]
    RF_voltage = config["V0max"] * 1e-3
    harm_number = config["h"]
    cc_name_knobs = config["cc_name_knobs"]  # MAD-X knobs for RF cavities
    bunch_intensity = config["bunch_intensity"]
    particle = config["particle"]
    sequence_name = config["sequence_name"]
    mass = config["mass"]
    radius = config["radius"]
    nc = config["charge"]

    # Sequence and beam
    madx.command.beam(particle=particle, energy=energy, mass=mass, charge=nc)
    madx.call(file=sequence)  # TODO: see above
    madx.use(sequence=sequence_name)
    madx.command.twiss()

    # Get some parameters from MAD-X Twiss
    twiss = madx.table.twiss.dframe()
    summ = madx.table.summ.dframe()
    RC = summ["length"].to_numpy()[0]  # ring circumference
    gt = summ["gammatr"].to_numpy()[0]  # transition energy gamma
    gamma = madx.table.twiss.summary.gamma  # relativistic gamma
    etap = abs(1.0 / gt**2 - 1.0 / gamma**2)  # ???
    betar = np.sqrt(1.0 - 1.0 / gamma**2)  # relativistic beta
    En = madx.table.twiss.summary.energy  # beam energy in [GeV]
    E0 = En / gamma  # rest mass energy in [GeV]

    madx.input(f"{cc_name_knobs}={RF_voltage*1e3}*{nc};")
    U0 = 0
    bunch_length = config["blns"] * 1e-9 / 4.0 * (c * betar)  # bunch length in [m]
    bunch_length_level = config["bl_lev"] * 1e-9 / 4.0 * (c * betar)  # ???
    bl_i = bunch_length
    exi = exn / (gamma * betar)  # geom emit x in [m]
    eyi = eyn / (gamma * betar)  # geom emit y in [m]

    frev = betar * c / RC  # revolution frequency in [Hz]
    fRF = harm_number * frev  # RF cavity frequency in [Hz]
    TRF = 1.0 / fRF  # RF period in [s]

    dpp = _bl_to_dpp(RC, En, nc, RF_voltage, U0, betar, harm_number, etap, bl_i)  # relative momentum spread
    dee = dpp * betar**2  # relative energy spread

    # Twiss sequence, this time with RF system on
    madx.use(sequence=sequence_name)
    madx.command.twiss()
    twiss = madx.table.twiss.dframe()

    # Set some properties for the beam
    madx.beam.ex = exi  # set the geom emit x (in [m])
    madx.beam.ey = eyi  # set the geom emit y (in [m])
    madx.beam.sigt = bunch_length  # set the bunch length (in [m])
    madx.beam.sige = dee  # set the relative energy spread
    madx.beam.npart = bunch_intensity  # number of particles per bunch

    # TODO: why are we setting these to 0?
    U0, taux, tauy, taul, ex0, ey0, sp0, ss0 = (0,) * 8
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
