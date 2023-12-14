"""
Tests in here check that some specific behaviours are implemented correctly in
the analytical classes for some edge cases.

For example, check the NagaitsevIBS class logs a message if the integrals have
not been computed when asking for the growth rates, but performs the calculation
of both anyway.

We also check for instance that the BjorkenMtingwaIBS issues a warning when it is
asked to compute IBS growth rates with a Twiss that was not centered.
"""
import logging

import pytest

from xibs.analytical import BjorkenMtingwaIBS, IBSGrowthRates, NagaitsevIBS, NagaitsevIntegrals
from xibs.inputs import BeamParameters, OpticsParameters


def test_nagaitsev_growth_rates_computes_integrals_if_absent(xtrack_ps_injection_protons, caplog):
    """
    Checking that NagaitsevIBS.growth_rates logs a message if the calculation
    of the integrals has not been performed beforehand, and then performs the
    calculation (also for the growth rates).
    """
    caplog.set_level(logging.INFO)
    # --------------------------------------------------------------------
    # Load xsuite line (PS here because it's smaller/faster) and init IBS
    line = xtrack_ps_injection_protons
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters(line.particle_ref)
    beamparams.n_part = int(8.1e8)  # value doesn't matter much
    IBS = NagaitsevIBS(beamparams, opticsparams)
    assert IBS.elliptic_integrals is None  # make sure it's not already there
    # --------------------------------------------------------------------
    # Check that .growth_rates logs and that the integrals + growth rates are computed
    # We use random values as they don't matter but need to specify compute_integrals=False
    IBS.growth_rates(2e-6, 2e-6, 1e-5, 1e-5, compute_integrals=False)

    record = caplog.records[0]  # check just the first logging message
    assert record.levelname == "INFO"
    assert "Computing growth rates requires having computed Nagaitsev integrals." in record.message
    assert "They will be computed first." in record.message

    assert IBS.elliptic_integrals is not None
    assert isinstance(IBS.elliptic_integrals, NagaitsevIntegrals)
    assert IBS.ibs_growth_rates is not None
    assert isinstance(IBS.ibs_growth_rates, IBSGrowthRates)


def test_nagaitsev_growth_rates_computes_integrals_by_default(xtrack_ps_injection_protons, caplog):
    """
    Checking that NagaitsevIBS.growth_rates updates the integrals by
    default before computing the growth rates.
    """
    caplog.set_level(logging.INFO)
    # --------------------------------------------------------------------
    # Load xsuite line (PS here because it's smaller/faster) and init IBS
    line = xtrack_ps_injection_protons
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters(line.particle_ref)
    beamparams.n_part = int(8.1e8)  # value doesn't matter much
    IBS = NagaitsevIBS(beamparams, opticsparams)
    assert IBS.elliptic_integrals is None  # make sure it's not already there
    # --------------------------------------------------------------------
    # We put fake values into the instance's .elliptic_integrals to check
    # afterwards that they are properly updated
    inital = NagaitsevIntegrals(Ix=1, Iy=1, Iz=1)
    IBS.elliptic_integrals = inital
    # --------------------------------------------------------------------
    # Check that .growth_rates logs and that the integrals + growth rates are computed
    # We use random values as they don't matter but need to specify compute_integrals=False
    IBS.growth_rates(2e-6, 2e-6, 1e-5, 1e-5)

    assert IBS.elliptic_integrals is not None
    assert isinstance(IBS.elliptic_integrals, NagaitsevIntegrals)
    assert IBS.elliptic_integrals != inital  # check it's not the same instance
    assert IBS.ibs_growth_rates is not None
    assert isinstance(IBS.ibs_growth_rates, IBSGrowthRates)


def test_bjorken_mtingwa_growth_rates_warns_if_twiss_not_centered(
    madx_ps_injection_protons, xtrack_ps_injection_protons, caplog
):
    """
    Checking that BjorkenMtingwaIBS.growth_rates issues a warning (and logs one) when asked to
    compute IBS growth rates with a Twiss that was not centered.
    """
    caplog.set_level(logging.WARNING)
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, _ = madx_ps_injection_protons  # fully set up from the config file
    # --------------------------------------------------------------------
    # Load the optics, from MAD-X to make sure it's not centered
    twiss = madx.twiss(centre=False).dframe()  # we explicitely don't want centered
    seq_name = madx.table.twiss.summary.sequence  # works whichever your sequence
    frev_hz = madx.sequence[seq_name].beam.freq0 * 1e6  # beware freq0 is in MHz
    gamma_rel = madx.sequence[seq_name].beam.gamma  # relativistic gamma
    gamma_tr = madx.table.summ.gammatr[0]  # transition gamma
    slipfactor = (1 / (gamma_tr**2)) - (1 / (gamma_rel**2))  # use the xsuite convention!
    opticsparams = OpticsParameters(twiss, slipfactor, frev_hz)
    # --------------------------------------------------------------------
    # Get BeamParameters - the values don't matter as we just test the warning
    beamparams = BeamParameters(xtrack_ps_injection_protons.particle_ref)
    # --------------------------------------------------------------------
    # Initialize IBS class and already assert opticsparams has detected twiss is not centered
    IBS = BjorkenMtingwaIBS(beamparams, opticsparams)
    assert IBS.optics._is_centered is False
    # --------------------------------------------------------------------
    # Check the warning is raised by .growth_rates
    with pytest.warns(match="The provided Twiss was calculated at the exit of the elements"):
        IBS.growth_rates(2e-6, 2e-6, 1e-5, 1e-5)  # random values as they don't matter
    assert isinstance(IBS.ibs_growth_rates, IBSGrowthRates)
    # --------------------------------------------------------------------
    # Also check the warning log message
    record = caplog.records[0]  # check just the first logging message
    assert record.levelname == "WARNING"
    assert "Twiss was not calculated at center of elements" in record.message
