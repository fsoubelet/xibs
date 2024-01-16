"""
Tests in here check that errors that should be raised by the kick classes
(SimpleKickIBS and KineticKickIBS) are indeed raised.
"""
import logging

import pytest

from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KickBasedIBS, KineticKickIBS, SimpleKickIBS


def test_simple_kick_raises_if_below_transition(madx_ps_injection_protons, caplog):
    """
    Checking that SimpleKickIBS initialization raises an error if it detects that the
    machine operates below transition, as it is not adapted for this.
    """
    caplog.set_level(logging.INFO)
    # --------------------------------------------------------------------
    # Get the inputs from MAD-X and initialize IBS class
    madx, params = madx_ps_injection_protons  # fully set up from the config file
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    assert opticsparams.slip_factor < 0  # make sure the test case is correct: below transition
    # --------------------------------------------------------------------
    # Check the error is raised by initilalization of SimpleKickIBS
    with pytest.raises(NotImplementedError):
        IBS = SimpleKickIBS(beamparams, opticsparams)

    for record in caplog.records:  # check the logging message
        assert record.levelname == "ERROR"
        assert (
            "The provided optics parameters indication that the machine is below transition,"
            in record.message
        )
        assert "which is incompatible with SimpleKickIBS (see documentation)" in record.message
        assert "Use the kinetic formalism with KineticKickIBS instead." in record.message


@pytest.mark.parametrize("IBSClass", [SimpleKickIBS, KineticKickIBS])
def test_apply_ibs_kick_raises_if_no_coefficients(xtrack_sps_injection_protons, IBSClass, caplog):
    """
    Checking that KickBasedIBS.apply_ibs_kick raises and error if the calculation
    of the kick coefficients has not been performed beforehand.
    """
    caplog.set_level(logging.ERROR)
    # --------------------------------------------------------------------
    # Load xsuite line (PS here because it's smaller/faster) and init IBS
    line = xtrack_sps_injection_protons
    opticsparams = OpticsParameters.from_line(line)
    beamparams = BeamParameters.from_line(line, n_part=8.1e8)  # n_part doesn't matter
    IBS: KickBasedIBS = IBSClass(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Check the error is raised by .emittance_evolution
    with pytest.raises(AttributeError):
        IBS.apply_ibs_kick(line.particle_ref)  # don't need a full distribution

    for record in caplog.records:  # check the logging message
        assert record.levelname == "ERROR"
        assert (
            "Attempted to apply IBS kick without having computed kick coefficients first." in record.message
        )
