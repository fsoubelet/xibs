"""
Tests in here check that errors that should be raised by the analytical classes
(BjorkenMtingwaIBS and NagaitsevIBS) are indeed raised.
"""
import pytest

from xibs.analytical import BjorkenMtingwaIBS, NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters


@pytest.mark.parametrize("IBSClass", [BjorkenMtingwaIBS, NagaitsevIBS])
def test_analytical_ibs_emittance_evolution_raises_if_no_growth_rates(
    xtrack_ps_injection_protons, IBSClass, caplog
):
    """
    Checking that NagaitsevIBS.emittance_evolution raises and error if the calculation
    of the growth rates has not been performed beforehand.
    """
    # --------------------------------------------------------------------
    # Load xsuite line (PS here because it's smaller/faster) and init IBS
    line = xtrack_ps_injection_protons
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters(line.particle_ref)
    beamparams.n_part = int(8.1e8)  # value doesn't matter much
    IBS = IBSClass(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Check the error is raised by .emittance_evolution
    with pytest.raises(ValueError):
        IBS.emittance_evolution(2e-6, 2e-6, 1e-5, 1e-5)  # random values as they don't matter

    for record in caplog.records:  # check the logging message
        assert record.levelname == "ERROR"
        assert (
            "Attempted to compute emittance evolution without having computed growth rates" in record.message
        )


def test_nagaitsev_raises_on_coasting_beams(xtrack_ps_injection_protons, caplog):
    """
    Checking that NagaitsevIBS.growth_rates raises a NotImplementedError
    if the user asks for coasting beams, and redirects to BjorkenMtingwaIBS.
    """
    # --------------------------------------------------------------------
    # Load xsuite line (PS here because it's smaller/faster) and init IBS
    line = xtrack_ps_injection_protons
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters(line.particle_ref)
    beamparams.n_part = int(8.1e8)  # value doesn't matter much
    IBS = NagaitsevIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Check the error is raised by .growth_rates with bunched=False
    with pytest.raises(NotImplementedError) as execinfo:
        IBS.growth_rates(2e-6, 2e-6, 1e-5, 1e-5, bunched=False)  # random values as they don't matter
    assert "Calculation for coasting beams is not implemented yet in this formalism." in str(execinfo.value)
    assert "Please use the BjorkenMtignwaIBS class instead, which supports this feature." in str(
        execinfo.value
    )
    # --------------------------------------------------------------------
    # Check the logged error message
    for record in caplog.records:
        assert record.levelname == "ERROR"
        assert (
            "Computing growth rates for coasting beams is currently not supported in this class."
            in record.message
        )
