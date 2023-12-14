"""
Tests in here are each for different LHC configurations (inj/top and protons/ions), but this time they include vertical dispersion.
To do so, the specific crossing knobs are not set to 0, but to their operational values.
We are computing the growth rates from MAD-X, and from the package's BjorkenMtingwaIBS implementation, then compare results to ensure they are consistent.

FOR COMPARISONS, PLEASE KEEP IN MIND:
    #  - We are using np.isclose so the reference number is the second argument
    #  - We have to explicitely set atol=0 as the numbers can be very small
"""
import numpy as np
import pytest

from helpers import get_madx_ibs_growth_rates

from xibs.analytical import BjorkenMtingwaIBS
from xibs.inputs import BeamParameters, OpticsParameters

# ----- Tests at injection energy ----- #


@pytest.mark.parametrize("bunched", [True, False])
def test_lhc_injection_protons(madx_lhc_injection_protons_with_vertical_disp, bunched):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, params = madx_lhc_injection_protons_with_vertical_disp  # fully set up from the config file
    madx.beam.bunched = bunched  # set bunched or coasting beam
    madx.command.twiss(centre=True)  # re-iterate Twiss before calling IBS
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # --------------------------------------------------------------------
    # Load the optics and beam params from MAD-X for maximum consistency
    # (also, we want to use a centered Twiss to get closer to MAD-X results)
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    # --------------------------------------------------------------------
    # Compute the growth rates
    IBS = BjorkenMtingwaIBS(beamparams, opticsparams)
    IBS.growth_rates(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length, bunched=bunched
    )
    # --------------------------------------------------------------------
    # Now we compare values to MAD-X
    assert np.isclose(IBS.ibs_growth_rates.Tx, mad_Tx, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Ty, mad_Ty, atol=0, rtol=5.5e-2)  # bit more margin in vertical
    assert np.isclose(IBS.ibs_growth_rates.Tz, mad_Tz, atol=0, rtol=1e-2)


# ----- Tests at top energy ----- #


@pytest.mark.parametrize("bunched", [True, False])
def test_lhc_top_protons(madx_lhc_top_protons_with_vertical_disp, bunched):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, params = madx_lhc_top_protons_with_vertical_disp  # fully set up from the config file
    madx.beam.bunched = bunched  # set bunched or coasting beam
    madx.command.twiss(centre=True)  # re-iterate Twiss before calling IBS
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # --------------------------------------------------------------------
    # Load the optics and beam params from MAD-X for maximum consistency
    # (also, we want to use a centered Twiss to get closer to MAD-X results)
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    # --------------------------------------------------------------------
    # Compute the growth rates
    IBS = BjorkenMtingwaIBS(beamparams, opticsparams)
    IBS.growth_rates(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length, bunched=bunched
    )
    # --------------------------------------------------------------------
    # Now we compare values to MAD-X
    assert np.isclose(IBS.ibs_growth_rates.Tx, mad_Tx, atol=0, rtol=1.5e-2)
    assert np.isclose(IBS.ibs_growth_rates.Ty, mad_Ty, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Tz, mad_Tz, atol=0, rtol=1e-2)


@pytest.mark.parametrize("bunched", [True, False])
def test_lhc_top_ions(madx_lhc_top_ions_with_vertical_disp, bunched):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, params = madx_lhc_top_ions_with_vertical_disp  # fully set up from the config file
    madx.beam.bunched = bunched  # set bunched or coasting beam
    madx.command.twiss(centre=True)  # re-iterate Twiss before calling IBS
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # --------------------------------------------------------------------
    # Load the optics and beam params from MAD-X for maximum consistency
    # (also, we want to use a centered Twiss to get closer to MAD-X results)
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    # --------------------------------------------------------------------
    # Compute the growth rates
    IBS = BjorkenMtingwaIBS(beamparams, opticsparams)
    IBS.growth_rates(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length, bunched=bunched
    )
    # --------------------------------------------------------------------
    # Now we compare values to MAD-X
    assert np.isclose(IBS.ibs_growth_rates.Tx, mad_Tx, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Ty, mad_Ty, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Tz, mad_Tz, atol=0, rtol=1e-2)
