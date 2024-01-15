"""
Tests in here check that some specific behaviours are implemented correctly in
the kick classes for some edge cases.

For example, check the initialization of SimpleKickIBS based on the detected vertical dispersion
in the machine.

We also check for instance that the BjorkenMtingwaIBS issues a warning when it is
asked to compute IBS growth rates with a Twiss that was not centered.

We also check that providing (equivalent) geometric or normalized emittances does
not change the results of the calculations.
"""
import logging

import numpy as np
import pytest

from xibs.analytical import BjorkenMtingwaIBS, NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import SimpleKickIBS


def test_simple_kick_chooses_bjorkenmtingwa_with_vertical_dispersion(
    madx_lhc_injection_protons_with_vertical_disp, caplog
):
    """
    Checking that SimpleKickIBS initialization logs a message about its choice of analytical
    formalism and makes the correct one when no vertical dispersion is in the machine, aka
    using NagaitsevIBS. Here we use LHC injection protons with xing angles MAD-X as it has
    vertical dispersion.
    """
    caplog.set_level(logging.ERROR)  # so we don't catch warnings from input parameters
    # --------------------------------------------------------------------
    # Get the inputs from MAD-X and initialize IBS class
    madx, params = madx_lhc_injection_protons_with_vertical_disp  # fully set up from the config file
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    caplog.set_level(logging.INFO)  # now let's capture the interesting messages
    IBS = SimpleKickIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Check that the chosen analytical formalism is Nagaitsev
    assert isinstance(IBS.analytical_ibs, BjorkenMtingwaIBS)
    # Check proper linking betweeb instance attributes
    assert IBS.beam_parameters is IBS.analytical_ibs.beam_parameters
    assert IBS.optics is IBS.analytical_ibs.optics
    # --------------------------------------------------------------------
    # Check that the choice has been logged to the user, with the possibility
    # of overriding it
    record = caplog.records[0]  # formalism choice info should be in the first logged message
    assert record.levelname == "INFO"
    assert (
        "Non-zero vertical dispersion detected in the lattice, using Bjorken & Mtingwa formalism"
        in record.message
    )
    record = caplog.records[-1]  # overriding possibility should be in the last logged message
    assert record.levelname == "INFO"
    assert (
        "This can be overridden manually, by explicitely setting the self.analytical_ibs attribute"
        in record.message
    )


def test_simple_kick_chooses_nagaitsev_without_vertical_dispersion(madx_sps_injection_protons, caplog):
    """
    Checking that SimpleKickIBS initialization logs a message about its choice of analytical
    formalism and makes the correct one when no vertical dispersion is in the machine, aka
    using NagaitsevIBS. We use SPS injection protons MAD-X as it has no vertical dispersion.
    """
    caplog.set_level(logging.ERROR)  # so we don't catch warnings from input parameters
    # --------------------------------------------------------------------
    # Get the inputs from MAD-X and initialize IBS class
    madx, params = madx_sps_injection_protons  # fully set up from the config file
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    caplog.set_level(logging.INFO)  # now let's capture the interesting messages
    IBS = SimpleKickIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Check that the chosen analytical formalism is Nagaitsev
    assert isinstance(IBS.analytical_ibs, NagaitsevIBS)
    # Check proper linking betweeb instance attributes
    assert IBS.beam_parameters is IBS.analytical_ibs.beam_parameters
    assert IBS.optics is IBS.analytical_ibs.optics
    # --------------------------------------------------------------------
    # Check that the choice has been logged to the user, with the possibility
    # of overriding it
    record = caplog.records[0]  # formalism choice info should be in the first logged message
    assert record.levelname == "INFO"
    assert "No vertical dispersion in the lattice, using Nagaitsev formalism" in record.message
    record = caplog.records[-1]  # overriding possibility should be in the last logged message
    assert record.levelname == "INFO"
    assert (
        "This can be overridden manually, by explicitely setting the self.analytical_ibs attribute"
        in record.message
    )
