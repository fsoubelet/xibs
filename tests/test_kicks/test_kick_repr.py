"""
Tests in here check that the REPRs of the kick classes work as intended.
"""
import warnings

import xtrack as xt

from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS, SimpleKickIBS


def test_simple_kick_repr(xtrack_sps_injection_protons):
    # --------------------------------------------------------------------
    # Get the BjorkenMtingwaIBS instance
    line: xt.Line = xtrack_sps_injection_protons
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters(line.particle_ref)  # npart doesn't matter here
    IBS = SimpleKickIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Check the repr (which calls __str__) withough having calculated integrals nor growth rates
    assert "SimpleKickIBS object for kick-based IBS calculations." in IBS.__repr__()
    assert "IBS kick coefficients computed: False" in IBS.__repr__()


def test_kinetic_kick_repr(xtrack_ps_injection_protons):
    # --------------------------------------------------------------------
    # Get the BjorkenMtingwaIBS instance
    line: xt.Line = xtrack_ps_injection_protons
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters(line.particle_ref)  # npart doesn't matter here
    IBS = KineticKickIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Check the repr (which calls __str__) withough having calculated growth rates
    assert "KineticKickIBS object for kick-based IBS calculations." in IBS.__repr__()
    assert "IBS kick coefficients computed: False" in IBS.__repr__()
