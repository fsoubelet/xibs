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
            "The provided optics parameters indicate that the machine is below transition," in record.message
        )
        assert "which is incompatible with SimpleKickIBS (see documentation)" in record.message
        assert "Use the kinetic formalism with KineticKickIBS instead." in record.message
