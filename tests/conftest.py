"""
Additional tools for testing, all fixtures defined here are discovered by and available to all tests automatically.
Throughout the tests, remember to only benchmark by comparing analytical quantities and not some evolution over
many tracked turns, as the kinetic or simple kicks are applied with a random component and it will mess up any
comparison.
"""
import pathlib

import pytest
import xtrack as xt

from cpymad.madx import Madx
from helpers import make_sps_thin, re_cycle_sequence

# ----- Paths to inputs ----- #

CURRENT_DIR = pathlib.Path(__file__).parent
INPUTS_DIR = CURRENT_DIR / "inputs"

# Various necessary files in acc-models-sps
ACC_MODELS_SPS = INPUTS_DIR / "acc-models-sps"  # .gitignored so can be kept locally, is cloned in our CI
SPS_SEQUENCE = ACC_MODELS_SPS / "SPS_LS2_2020-05-26.seq"
SPS_TOOLKIT = ACC_MODELS_SPS / "toolkit"
SPS_LHC_IONS_OPTICS = ACC_MODELS_SPS / "strengths" / "lhc_ion.str"
SPS_LHC_IONS_BEAMS = ACC_MODELS_SPS / "beams" / "beam_lhc_ion_injection.madx"

# Files for CLIC Damping Ring used in Michail's scripts
CLIC_DR_LINE_JSON = INPUTS_DIR / "chrom-corr_DR.newlattice_2GHz.json"
CLIC_DR_SEQUENCE_MADX = INPUTS_DIR / "chrom-corr_DR.newlattice_2GHz.seq"

# ----- Lines and Sequences Fixtures ----- #


@pytest.fixture()
def matched_sps_lhc_ions_injection() -> Madx:
    """
    A cpymad.Madx instance with loaded SPS sequence, lhc ions optics,
    and matched parameters. It is a thin lattice.
    """
    with Madx(stdout=False) as madx:
        # Parameters for matching later on
        qx, qy, dqx, dqy = 26.30, 26.25, -3.0e-9, -3.0e-9

        # Call sequence, optics and define beams
        madx.call(str(SPS_SEQUENCE.absolute()))
        madx.call(str(SPS_LHC_IONS_OPTICS.absolute()))
        madx.call(str(SPS_LHC_IONS_BEAMS.absolute()))
        madx.command.use(sequence="sps")
        madx.command.twiss()

        # Makethin, call some definition macros
        re_cycle_sequence(madx)  # TODO: could use cpymadtools for this
        madx.command.use(sequence="sps")
        make_sps_thin(madx, sequence="sps", slicefactor=5)
        madx.command.use(sequence="sps")
        madx.call(str((SPS_TOOLKIT / "macro.madx").absolute()))
        madx.exec(f"sps_match_tunes({qx},{qy});")  # TODO: could use cpymadtools for this
        madx.exec("sps_define_sext_knobs();")
        madx.exec("sps_set_chroma_weights_q26();")

        # Match chromas (TODO: could use cpymadtools for this)
        madx.command.match()
        madx.command.global_(dq1=dqx)
        madx.command.global_(dq2=dqy)
        madx.command.vary(name="qph_setvalue")
        madx.command.vary(name="qpv_setvalue")
        madx.command.jacobian(calls=50, tolerance=1e-25)
        madx.command.endmatch()

        # Yield, exits context manager only after the calling test is done
        yield madx


@pytest.fixture()
def xsuite_line_CLIC_damping_ring() -> xt.Line:
    """
    A loaded xt.Line of the CLIC DR with chroma corrected, as used in
    scripts from Michail to benchmark against.
    """
    # Load the line
    line = xt.Line.from_json(str(CLIC_DR_LINE_JSON.absolute()))
    # Simplify the line
    line.remove_inactive_multipoles(inplace=True)
    line.remove_zero_length_drifts(inplace=True)
    line.merge_consecutive_drifts(inplace=True)
    line.merge_consecutive_multipoles(inplace=True)
    # Build tracker (default context)
    line.build_tracker(extra_headers=["#define XTRACK_MULTIPOLE_NO_SYNRAD"])
    # Activate the cavities
    for cavity in [element for element in line.elements if isinstance(element, xt.Cavity)]:
        cavity.lag = 180
    return line


@pytest.fixture()
def madx_CLIC_damping_ring() -> Madx:
    """
    A cpymad.Madx instance with loaded CLIC DR sequence file and optics,
    as used in scripts from Michail to benchmark against. It is a thin
    lattice.
    """
    with Madx(stdout=False) as madx:
        madx = Madx(stdout=False)
        madx.call(str(CLIC_DR_SEQUENCE_MADX.absolute()))
        # Makethin on RING sequence
        n_slice_per_element = 4
        madx.command.beam(particle="positron", energy=2.86, bunched=True)
        madx.command.use(sequence="RING")
        madx.command.select(flag="MAKETHIN", slice_=n_slice_per_element, thick=False)
        madx.command.select(flag="MAKETHIN", pattern="wig", slice_=1)
        madx.command.makethin(sequence="RING", makedipedge=True)
        madx.command.use(sequence="RING")

        # Yield, exits context manager only after the calling test is done
        yield madx
