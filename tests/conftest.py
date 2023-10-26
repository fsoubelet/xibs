"""
Additional tools for testing, all fixtures defined here are discovered by and available to all tests automatically.
Throughout the tests, remember to only benchmark by comparing analytical quantities and not some evolution over
many tracked turns, as the kinetic or simple kicks are applied with a random component and it will mess up any
comparison.
"""
import pathlib

from typing import Dict

import pytest
import xpart as xp
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

# -- LHC fixtures -- #


@pytest.fixture(scope="function")
def madx_lhc_injection_protons() -> Madx:
    """
    A cpymad.Madx instance with loaded LHCB1 sequence, protons at
    injection energy.
    """
    with open(CONFIGS_DIR / "lhc_injection_protons.yaml") as config_file:
        config = yaml.safe_load(config_file)

    with Madx(stdout=False) as madx:
        params = setup_madx_from_config(madx, config)
        yield madx, params


# No injection ions??


@pytest.fixture(scope="function")
def madx_lhc_top_protons() -> Madx:
    """
    A cpymad.Madx instance with loaded LHCB1 sequence, protons at
    top energy.
    """
    with open(CONFIGS_DIR / "lhc_top_protons.yaml") as config_file:
        config = yaml.safe_load(config_file)

    with Madx(stdout=False) as madx:
        params = setup_madx_from_config(madx, config)
        yield madx, params


@pytest.fixture(scope="function")
def madx_lhc_top_ions() -> Madx:
    """
    A cpymad.Madx instance with loaded LHCB1 sequence, ions at
    top energy.
    """
    with open(CONFIGS_DIR / "lhc_top_ions.yaml") as config_file:
        config = yaml.safe_load(config_file)

    with Madx(stdout=False) as madx:
        params = setup_madx_from_config(madx, config)
        yield madx, params


# -- SPS fixtures -- #


@pytest.fixture(scope="function")
def madx_sps_injection_protons() -> Madx:
    """
    A cpymad.Madx instance with loaded SPS sequence, protons at
    injection energy (Q26 configuration).
    """
    with open(CONFIGS_DIR / "sps_injection_protons.yaml") as config_file:
        config = yaml.safe_load(config_file)

    with Madx(stdout=False) as madx:
        params = setup_madx_from_config(madx, config)
        yield madx, params


@pytest.fixture(scope="function")
def madx_sps_injection_ions() -> Madx:
    """
    A cpymad.Madx instance with loaded SPS sequence, ions at
    injection energy.
    """
    with open(CONFIGS_DIR / "sps_injection_ions.yaml") as config_file:
        config = yaml.safe_load(config_file)

    with Madx(stdout=False) as madx:
        params = setup_madx_from_config(madx, config)
        yield madx, params


@pytest.fixture(scope="function")
def madx_sps_top_protons() -> Madx:
    """
    A cpymad.Madx instance with loaded SPS sequence, protons at
    top energy (Q26 configuration).
    """
    with open(CONFIGS_DIR / "sps_top_protons.yaml") as config_file:
        config = yaml.safe_load(config_file)

    with Madx(stdout=False) as madx:
        params = setup_madx_from_config(madx, config)
        yield madx, params


@pytest.fixture(scope="function")
def madx_sps_top_ions() -> Madx:
    """
    A cpymad.Madx instance with loaded SPS sequence, ions at
    top energy.
    """
    with open(CONFIGS_DIR / "sps_top_ions.yaml") as config_file:
        config = yaml.safe_load(config_file)

    with Madx(stdout=False) as madx:
        params = setup_madx_from_config(madx, config)
        yield madx, params


# -- PS fixtures -- #


@pytest.fixture(scope="function")
def madx_ps_injection_protons() -> Madx:
    """
    A cpymad.Madx instance with loaded PS sequence, protons at
    injection energy.
    """
    with open(CONFIGS_DIR / "ps_injection_protons.yaml") as config_file:
        config = yaml.safe_load(config_file)

    with Madx(stdout=False) as madx:
        params = setup_madx_from_config(madx, config)
        yield madx, params


@pytest.fixture(scope="function")
def madx_ps_injection_ions() -> Madx:
    """
    A cpymad.Madx instance with loaded PS sequence, ions at
    injection energy.
    """
    with open(CONFIGS_DIR / "ps_injection_ions.yaml") as config_file:
        config = yaml.safe_load(config_file)

    with Madx(stdout=False) as madx:
        params = setup_madx_from_config(madx, config)
        yield madx, params


# ----- Xtrack Lines Fixtures ----- #

# -- LHC fixtures -- #


@pytest.fixture(scope="function")
def xtrack_lhc_injection_protons() -> xt.Line:
    """An `xtrack.Line` of the LHCB1 sequence for protons at injection energy."""
    line_json = LINES_DIR / "lhc_injection_protons.json"
    return xt.Line.from_json(line_json)


# No injection ions??


@pytest.fixture(scope="function")
def xtrack_lhc_top_protons() -> xt.Line:
    """An `xtrack.Line` of the LHCB1 sequence for protons at top energy."""
    line_json = LINES_DIR / "lhc_top_protons.json"
    return xt.Line.from_json(line_json)


@pytest.fixture(scope="function")
def xtrack_lhc_top_ions() -> xt.Line:
    """An `xtrack.Line` of the LHCB1 sequence for ions at top energy."""
    line_json = LINES_DIR / "lhc_top_ions.json"
    return xt.Line.from_json(line_json)


# -- SPS fixtures -- #


@pytest.fixture(scope="function")
def xtrack_sps_injection_protons() -> xt.Line:
    """An `xtrack.Line` of the SPS sequence for protons at injection energy (Q26 configuration)."""
    line_json = LINES_DIR / "sps_injection_protons.json"
    return xt.Line.from_json(line_json)


@pytest.fixture(scope="function")
def xtrack_sps_injection_ions() -> xt.Line:
    """An `xtrack.Line` of the SPS sequence for ions at injection energy."""
    line_json = LINES_DIR / "sps_injection_ions.json"
    return xt.Line.from_json(line_json)


@pytest.fixture(scope="function")
def xtrack_sps_top_protons() -> xt.Line:
    """An `xtrack.Line` of the SPS sequence for protons at top energy (Q26 configuration)."""
    line_json = LINES_DIR / "sps_top_protons.json"
    return xt.Line.from_json(line_json)


@pytest.fixture(scope="function")
def xtrack_sps_top_ions() -> xt.Line:
    """An `xtrack.Line` of the SPS sequence for ions at top energy."""
    line_json = LINES_DIR / "sps_top_ions.json"
    return xt.Line.from_json(line_json)


# -- PS fixtures -- #


@pytest.fixture(scope="function")
def xtrack_ps_injection_protons() -> xt.Line:
    """An `xtrack.Line` of the PS sequence for protons at injection energy."""
    line_json = LINES_DIR / "ps_injection_protons.json"
    return xt.Line.from_json(line_json)


@pytest.fixture(scope="function")
def xtrack_ps_injection_ions() -> xt.Line:
    """An `xtrack.Line` of the PS sequence for ions at injection energy."""
    line_json = LINES_DIR / "ps_injection_ions.json"
    return xt.Line.from_json(line_json)


# ----- Private Utilities ----- #


def _make_xtrack_line_for_config(config: Dict, p0: xp.Particles) -> xt.Line:
    """
    Make an `xtrack.Line` equivalent to the MAD-X setup for a given config.
    Used to save the JSON files in `tests/inputs/lines` that are returned
    by the fixtures.

    Args:
        config (dict): the loaded yaml config file for the MAD-X setup.
        p0 (xp.Particles): the reference particle for the line.
    """
    with Madx() as madx:
        setup_madx_from_config(madx, config)
        seqname = config["sequence_name"]
        line = xt.Line.from_madx_sequence(madx.sequence[seqname], allow_thick=True)
        line.particle_ref = p0  # will be saved in the json file
        return line
