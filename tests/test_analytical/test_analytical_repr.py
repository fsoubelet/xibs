"""
Tests in here check that the REPRs of the analytical classes work as intended.
"""
import warnings

import xtrack as xt

from xibs.analytical import BjorkenMtingwaIBS, NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters


def test_nagaitsev_repr(xtrack_ps_injection_protons):
    # --------------------------------------------------------------------
    # Get the BjorkenMtingwaIBS instance
    line: xt.Line = xtrack_ps_injection_protons
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters(line.particle_ref)  # npart doesn't matter here
    IBS = NagaitsevIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Check the repr (which calls __str__) withough having calculated integrals nor growth rates
    assert "NagaitsevIBS object for analytical IBS calculations." in IBS.__repr__()
    assert "IBS growth rates computed: False" in IBS.__repr__()
    # --------------------------------------------------------------------
    # Check the repr after computing the growth rates
    IBS.growth_rates(2e-6, 2e-6, 1e-5, 1e-5)  # random values as they don't matter
    assert "IBS growth rates computed: True" in IBS.__repr__()


def test_bjorken_mtingwa_repr(xtrack_ps_injection_protons):
    # --------------------------------------------------------------------
    # Get the BjorkenMtingwaIBS instance
    line: xt.Line = xtrack_ps_injection_protons
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters(line.particle_ref)  # npart doesn't matter here
    IBS = BjorkenMtingwaIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Check the repr (which calls __str__) withough having calculated growth rates
    assert "BjorkenMtingwaIBS object for analytical IBS calculations." in IBS.__repr__()
    assert "IBS growth rates computed: False" in IBS.__repr__()
    # --------------------------------------------------------------------
    # Check the repr after computing the growth rates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        IBS.growth_rates(2e-6, 2e-6, 1e-5, 1e-5)  # random values as they don't matter
    assert "IBS growth rates computed: True" in IBS.__repr__()
