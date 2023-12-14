"""
Tests in here are each for different PS configurations (injection protons/ions).
We are computing the growth rates from MAD-X, and from the package's BjorkenMtingwaIBS implementation, then compare results to ensure they are consistent.

FOR COMPARISONS, PLEASE KEEP IN MIND:
    #  - We are using np.isclose so the reference number is the second argument
    #  - We have to explicitely set atol=0 as the numbers can be very small
"""
import numpy as np

from helpers import get_madx_ibs_growth_rates

from xibs.analytical import BjorkenMtingwaIBS
from xibs.inputs import BeamParameters, OpticsParameters

# ----- Tests at injection energy ----- #


def test_ps_injection_protons(madx_ps_injection_protons):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, params = madx_ps_injection_protons  # fully set up from the config file
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # --------------------------------------------------------------------
    # Load the optics and beam params from MAD-X for maximum consistency
    # (also, we want to use a centered Twiss to get closer to MAD-X results)
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    # --------------------------------------------------------------------
    # Compute the growth rates
    IBS = BjorkenMtingwaIBS(beamparams, opticsparams)
    IBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    # --------------------------------------------------------------------
    # Now we compare values to MAD-X
    assert np.isclose(IBS.ibs_growth_rates.Tx, mad_Tx, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Ty, mad_Ty, atol=0, rtol=2.1e-2)  # bit more tolerance in y
    assert np.isclose(IBS.ibs_growth_rates.Tz, mad_Tz, atol=0, rtol=1e-2)


def test_ps_injection_ions(madx_ps_injection_ions):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, params = madx_ps_injection_ions  # fully set up from the config file
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # --------------------------------------------------------------------
    # Load the optics and beam params from MAD-X for maximum consistency
    # (also, we want to use a centered Twiss to get closer to MAD-X results)
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    # --------------------------------------------------------------------
    # Compute the growth rates
    IBS = BjorkenMtingwaIBS(beamparams, opticsparams)
    IBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    # --------------------------------------------------------------------
    # Now we compare values to MAD-X
    assert np.isclose(IBS.ibs_growth_rates.Tx, mad_Tx, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Ty, mad_Ty, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Tz, mad_Tz, atol=0, rtol=1e-2)
