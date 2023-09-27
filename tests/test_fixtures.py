"""
Only to test for now.
"""
import pytest

from helpers import get_madx_ibs_beam_size_growth_time


def test_sps_lhc_ions_injection_fixture(matched_sps_lhc_ions_injection):
    madx = matched_sps_lhc_ions_injection
    madx.command.twiss()  # needs to be called before IBS!
    Tx, Ty, Tl = get_madx_ibs_beam_size_growth_time(madx)
    print(Tx, Ty, Tl)
