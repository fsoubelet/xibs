"""
Additional tools for testing, all fixtures defined here are discovered by and available to all tests automatically.
Throughout the tests, remember to only benchmark by comparing analytical quantities and not some evolution over
many tracked turns, as the kinetic or simple kicks are applied with a random component and it will mess up any
comparison.
"""
import pathlib

import pytest
import xtrack as xt
import yaml

from cpymad.madx import Madx
from helpers import setup_madx_from_config

# ----- Paths to inputs ----- #

CURRENT_DIR = pathlib.Path(__file__).parent
INPUTS_DIR = CURRENT_DIR / "inputs"

# Various necessary files in acc-models-sps
ACC_MODELS_SPS = INPUTS_DIR / "acc-models-sps"  # .gitignored so can be kept locally, is cloned in our CI
SPS_SEQUENCE = ACC_MODELS_SPS / "SPS_LS2_2020-05-26.seq"
SPS_TOOLKIT = ACC_MODELS_SPS / "toolkit"
SPS_LHC_IONS_OPTICS = ACC_MODELS_SPS / "strengths" / "lhc_ion.str"
SPS_LHC_IONS_BEAMS = ACC_MODELS_SPS / "beams" / "beam_lhc_ion_injection.madx"

# Locations of folders with specific files for tests
CONFIGS_DIR = INPUTS_DIR / "configs"  # config files for MAD-X setups
LINES_DIR = INPUTS_DIR / "lines"  # (equivalent) frozen and saved xtrack.Lines

# ----- MAD-X Sequences Fixtures ----- #


@pytest.fixture()
def madx_lhc_injection_protons_no_crossing() -> Madx:
    """
    A cpymad.Madx instance with loaded LHC sequence, protons at
    injection energy,
    """
    with open(CONFIGS_DIR / "lhc_injection_protons_no_crossing.yaml") as config_file:
        config = yaml.safe_load(config_file)

    with Madx(stdout=False) as madx:
        # This will set up sequence, beam and RF parameters
        setup_madx_from_config(madx, config)

        # Yield, exits context manager only after the calling test is done
        yield madx


# ----- Xtrack Lines Fixtures ----- #
