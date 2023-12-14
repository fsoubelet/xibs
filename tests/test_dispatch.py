"""
Quick tests for the dispatch function.
"""
import pytest

from xibs.analytical import BjorkenMtingwaIBS, NagaitsevIBS
from xibs.dispatch import ibs
from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS, SimpleKickIBS


@pytest.mark.parametrize(
    "formalism", ["bjorken-mtingwa", "BJORKEN-MTINGWA", "Bjorken-Mtingwa", "b&m", "B&M", "B&m"]
)
def test_dispatch_BjorkenMtingwaIBS(formalism, xtrack_ps_injection_protons):
    # --------------------------------------------------------------------
    # Get beam and optics params from the xtrack Line
    line = xtrack_ps_injection_protons
    beam_params = BeamParameters(line.particle_ref)
    optics_params = OpticsParameters(line.twiss(method="4d"))
    # --------------------------------------------------------------------
    # Dispatch for BjorkenMtingwaIBS and check that proper class is returned
    # (and properly initialized)
    IBS = ibs(beam_params, optics_params, formalism)
    assert isinstance(IBS, BjorkenMtingwaIBS)
    assert IBS.beam_parameters == beam_params
    assert IBS.optics == optics_params


@pytest.mark.parametrize("formalism", ["nagaitsev", "NAGAITSEV", "Nagaitsev", "nAgaItSEv"])
def test_dispatch_NagaitsevIBS(formalism, xtrack_ps_injection_protons):
    # --------------------------------------------------------------------
    # Get beam and optics params from the xtrack Line
    line = xtrack_ps_injection_protons
    beam_params = BeamParameters(line.particle_ref)
    optics_params = OpticsParameters(line.twiss(method="4d"))
    # --------------------------------------------------------------------
    # Dispatch for NagaitsevIBS and check that proper class is returned
    # (and properly initialized)
    IBS = ibs(beam_params, optics_params, formalism)
    assert isinstance(IBS, NagaitsevIBS)
    assert IBS.beam_parameters == beam_params
    assert IBS.optics == optics_params


@pytest.mark.parametrize("formalism", ["kinetic", "KINETIC", "kINetiC", "kinETic"])
def test_dispatch_KineticKickIBS(formalism, xtrack_ps_injection_protons):
    # --------------------------------------------------------------------
    # Get beam and optics params from the xtrack Line
    line = xtrack_ps_injection_protons
    beam_params = BeamParameters(line.particle_ref)
    optics_params = OpticsParameters(line.twiss(method="4d"))
    # --------------------------------------------------------------------
    # Dispatch for KineticKickIBS and check that proper class is returned
    # (and properly initialized)
    IBS = ibs(beam_params, optics_params, formalism)
    assert isinstance(IBS, KineticKickIBS)
    assert IBS.beam_parameters == beam_params
    assert IBS.optics == optics_params


@pytest.mark.parametrize("formalism", ["simple", "SIMPLE", "sIMPLe", "siMpLe"])
def test_dispatch_SimpleKickIBS(formalism, xtrack_ps_injection_protons):
    # --------------------------------------------------------------------
    # Get beam and optics params from the xtrack Line
    line = xtrack_ps_injection_protons
    beam_params = BeamParameters(line.particle_ref)
    optics_params = OpticsParameters(line.twiss(method="4d"))
    # --------------------------------------------------------------------
    # Dispatch for SimpleKickIBS and check that proper class is returned
    # (and properly initialized)
    IBS = ibs(beam_params, optics_params, formalism)
    assert isinstance(IBS, SimpleKickIBS)
    assert IBS.beam_parameters == beam_params
    assert IBS.optics == optics_params


@pytest.mark.parametrize("formalism", ["wrong", "invalid", "nope", "notIBS"])
def test_dispatch_raises_on_invalid_formalism(formalism, caplog):
    # --------------------------------------------------------------------
    # No need for beam and optics params as the function raises before that
    with pytest.raises(ValueError):
        IBS = ibs(None, None, formalism)

    for record in caplog.records:  # check the logging message
        assert record.levelname == "ERROR"
        assert "Invalid formalism" in record.message
