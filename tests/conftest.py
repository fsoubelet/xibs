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


# -- LHC fixtures with crossing angles (and vertical dispersion) -- #


@pytest.fixture(scope="function")
def madx_lhc_injection_protons_with_vertical_disp() -> Madx:
    """
    A cpymad.Madx instance with loaded LHCB1 sequence, protons at
    injection energy.
    """
    with open(CONFIGS_DIR / "lhc_injection_protons.yaml") as config_file:
        config = yaml.safe_load(config_file)

    with Madx(stdout=False) as madx:
        # important: set remove_crossing_angles to False to have vertical dispersion
        params = setup_madx_from_config(madx, config, remove_crossing_angles=False)
        # The opticsfile calls toolkit/reset-bump-flags which sets all crossing flags to 0
        # anyway so I call the strengths again here explicitly
        madx.call(file="acc-models-lhc/strengths/ATS_Nominal/2023/ats_10m.madx")
        madx.command.twiss(centre=True)  # need to update twiss table for IBS command
        yield madx, params


@pytest.fixture(scope="function")
def madx_lhc_top_protons_with_vertical_disp() -> Madx:
    """
    A cpymad.Madx instance with loaded LHCB1 sequence, protons at
    top energy.
    """
    with open(CONFIGS_DIR / "lhc_top_protons.yaml") as config_file:
        config = yaml.safe_load(config_file)

    with Madx(stdout=False) as madx:
        # important: set remove_crossing_angles to False to have vertical dispersion
        params = setup_madx_from_config(madx, config, remove_crossing_angles=False)
        # The opticsfile calls toolkit/reset-bump-flags which sets all crossing flags to 0
        # anyway so I call the strengths again here explicitly
        madx.call(file="acc-models-lhc/strengths/ATS_Nominal/2023/ats_30cm.madx")
        madx.command.twiss(centre=True)  # need to update twiss table for IBS command
        yield madx, params


@pytest.fixture(scope="function")
def madx_lhc_top_ions_with_vertical_disp() -> Madx:
    """
    A cpymad.Madx instance with loaded LHCB1 sequence, ions at
    top energy.
    """
    with open(CONFIGS_DIR / "lhc_top_ions.yaml") as config_file:
        config = yaml.safe_load(config_file)

    with Madx(stdout=False) as madx:
        # important: set remove_crossing_angles to False to have vertical dispersion
        params = setup_madx_from_config(madx, config, remove_crossing_angles=False)
        # The opticsfile calls toolkit/reset-bump-flags which sets all crossing flags to 0
        # anyway so I call the strengths again here explicitly
        madx.call(file="acc-models-lhc/strengths/ATS_Nominal/2023_IONS/50cm.madx")
        madx.command.twiss(centre=True)  # need to update twiss table for IBS command
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


# -- CLIC DR fixture -- #

@pytest.fixture(scope="function")
def xtrack_clic_damping_ring() -> xt.Line:
    """An `xtrack.Line` of the CLIC DR for positrons."""
    line_json = LINES_DIR / "clic_damping_ring.json"
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
