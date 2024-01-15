"""
Tests in here check that some specific behaviours are implemented correctly in
the analytical classes for some edge cases.

For example, check the NagaitsevIBS class logs a message if the integrals have
not been computed when asking for the growth rates, but performs the calculation
of both anyway.

We also check for instance that the BjorkenMtingwaIBS issues a warning when it is
asked to compute IBS growth rates with a Twiss that was not centered.

We also check that providing (equivalent) geometric or normalized emittances does
not change the results of the calculations.
"""
import logging

import numpy as np
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


def test_nagaitsev_integrals_equivalent_with_geometric_and_normalized_emittances(madx_ps_injection_protons):
    """
    Checking that NagaitsevIBS.integrals returns the same result when providing
    (equivalent) geometric or normalized emittances.
    """
    # --------------------------------------------------------------------
    # Get the inputs from MAD-X and initialize IBS class
    madx, params = madx_ps_injection_protons  # fully set up from the config file
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    IBS = NagaitsevIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Get the integrals with geometric emittances
    integrals_from_geom = IBS.integrals(params.geom_epsx, params.geom_epsy, params.sig_delta)
    # --------------------------------------------------------------------
    # Get the integrals with normalized emittances
    norm_emit_x = params.geom_epsx * beamparams.beta_rel * beamparams.gamma_rel
    norm_emit_y = params.geom_epsy * beamparams.beta_rel * beamparams.gamma_rel
    integrals_from_norm = IBS.integrals(
        norm_emit_x, norm_emit_y, params.sig_delta, normalized_emittances=True
    )
    # --------------------------------------------------------------------
    # Check the results are the same
    assert integrals_from_geom == integrals_from_norm


def test_nagaitsev_growth_rates_equivalent_with_geometric_and_normalized_emittances(
    madx_ps_injection_protons,
):
    """
    Checking that NagaitsevIBS.growth_rates returns the same result when providing
    (equivalent) geometric or normalized emittances.
    """
    # --------------------------------------------------------------------
    # Get the inputs from MAD-X and initialize IBS class
    madx, params = madx_ps_injection_protons  # fully set up from the config file
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    IBS = NagaitsevIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Get the growth rates with geometric emittances
    rates_from_geom = IBS.growth_rates(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length
    )
    # --------------------------------------------------------------------
    # Get the growth rates with normalized emittances
    norm_emit_x = params.geom_epsx * beamparams.beta_rel * beamparams.gamma_rel
    norm_emit_y = params.geom_epsy * beamparams.beta_rel * beamparams.gamma_rel
    rates_from_norm = IBS.growth_rates(
        norm_emit_x, norm_emit_y, params.sig_delta, params.bunch_length, normalized_emittances=True
    )
    # --------------------------------------------------------------------
    # Check the results are the same
    assert rates_from_geom == rates_from_norm


def test_bjorken_mtingwa_growth_rates_equivalent_with_geometric_and_normalized_emittances(
    madx_ps_injection_protons,
):
    """
    Checking that BjorkenMtingwaIBS.growth_rates returns the same result when providing
    (equivalent) geometric or normalized emittances.
    """
    # --------------------------------------------------------------------
    # Get the inputs from MAD-X and initialize IBS class
    madx, params = madx_ps_injection_protons  # fully set up from the config file
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    IBS = BjorkenMtingwaIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Get the growth rates with geometric emittances
    rates_from_geom = IBS.growth_rates(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length
    )
    # --------------------------------------------------------------------
    # Get the growth rates with normalized emittances
    norm_emit_x = params.geom_epsx * beamparams.beta_rel * beamparams.gamma_rel
    norm_emit_y = params.geom_epsy * beamparams.beta_rel * beamparams.gamma_rel
    rates_from_norm = IBS.growth_rates(
        norm_emit_x, norm_emit_y, params.sig_delta, params.bunch_length, normalized_emittances=True
    )
    # --------------------------------------------------------------------
    # Check the results are the same
    assert rates_from_geom == rates_from_norm


def test_analytical_emittance_evolution_equivalent_with_geometric_and_normalized_emittances(
    madx_ps_injection_protons,
):
    """
    Checking that BjorkenMtingwaIBS.growth_rates returns the same result when providing
    (equivalent) geometric or normalized emittances.
    """
    # --------------------------------------------------------------------
    # Get the inputs from MAD-X and initialize IBS class
    madx, params = madx_ps_injection_protons  # fully set up from the config file
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    MIBS = BjorkenMtingwaIBS(beamparams, opticsparams)
    NIBS = NagaitsevIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Get the growth rates for both
    MIBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    NIBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    # --------------------------------------------------------------------
    # Get equivalent normalized emittances from those in params
    norm_emit_x = params.geom_epsx * beamparams.beta_rel * beamparams.gamma_rel
    norm_emit_y = params.geom_epsy * beamparams.beta_rel * beamparams.gamma_rel
    # --------------------------------------------------------------------
    # Get the new_emittances with geometric emittances for both and check equality
    new_emitx_geom_bm, new_emity_geom_bm, _, _ = MIBS.emittance_evolution(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length
    )
    new_emitx_geom_nag, new_emity_geom_nag, _, _ = NIBS.emittance_evolution(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length
    )
    assert np.isclose(new_emitx_geom_bm, new_emitx_geom_nag, atol=0, rtol=1e-3)  # allow 0.1% deviation
    assert np.isclose(new_emity_geom_bm, new_emity_geom_nag, atol=0, rtol=1e-3)  # allow 0.1% deviation
    # --------------------------------------------------------------------
    # Get the new_emittances with normalized emittances for both and check equality
    new_emitx_norm_bm, new_emity_norm_bm, _, _ = MIBS.emittance_evolution(
        norm_emit_x, norm_emit_y, params.sig_delta, params.bunch_length, normalized_emittances=True
    )
    new_emitx_norm_nag, new_emity_norm_nag, _, _ = NIBS.emittance_evolution(
        norm_emit_x, norm_emit_y, params.sig_delta, params.bunch_length, normalized_emittances=True
    )
    assert np.isclose(new_emitx_norm_bm, new_emitx_norm_nag, atol=0, rtol=1e-3)  # allow 0.1% deviation
    assert np.isclose(new_emity_norm_bm, new_emity_norm_nag, atol=0, rtol=1e-3)  # allow 0.1% deviation
    # --------------------------------------------------------------------
    # Check that the equivalence of all returned results: geom <-> norm conversion still holds
    assert np.isclose(
        new_emitx_geom_bm * beamparams.beta_rel * beamparams.gamma_rel, new_emitx_norm_bm, atol=0, rtol=1e-3
    )  # allow 0.1% deviation
    assert np.isclose(
        new_emity_geom_bm * beamparams.beta_rel * beamparams.gamma_rel, new_emity_norm_bm, atol=0, rtol=1e-3
    )  # allow 0.1% deviation


def test_coulomb_log_equivalent_with_geometric_and_normalized_emittances(
    madx_ps_injection_protons,
):
    """
    Checking that BjorkenMtingwaIBS.growth_rates returns the same result when providing
    (equivalent) geometric or normalized emittances.
    """
    # --------------------------------------------------------------------
    # Get the inputs from MAD-X and initialize IBS class
    madx, params = madx_ps_injection_protons  # fully set up from the config file
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    IBS = BjorkenMtingwaIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Get the Coulomb log with geometric emittances
    coulog_from_geom = IBS.coulomb_log(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length
    )
    # --------------------------------------------------------------------
    # Get the Coulomb log with normalized emittances
    norm_emit_x = params.geom_epsx * beamparams.beta_rel * beamparams.gamma_rel
    norm_emit_y = params.geom_epsy * beamparams.beta_rel * beamparams.gamma_rel
    coulog_from_norm = IBS.coulomb_log(
        norm_emit_x, norm_emit_y, params.sig_delta, params.bunch_length, normalized_emittances=True
    )
    # --------------------------------------------------------------------
    # Check the results are the same
    assert coulog_from_geom == coulog_from_norm


def test_nagaitsev_warns_on_coasting_beams(xtrack_ps_injection_protons, caplog):
    """
    Checking that NagaitsevIBS.growth_rates logs a warning when provided with `bunch=False`
    (coasting beam) before computing the growth rates.
    """
    # --------------------------------------------------------------------
    # Load xsuite line (PS here because it's smaller/faster) and get IBS growth rates
    line = xtrack_ps_injection_protons
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters(line.particle_ref)
    beamparams.n_part = int(8.1e8)  # value doesn't matter much
    IBS = NagaitsevIBS(beamparams, opticsparams)
    IBS.growth_rates(2e-6, 2e-6, 1e-5, 1e-5, bunched=False)  # random values as they don't matter
    # --------------------------------------------------------------------
    # Check the logged warning message
    for record in caplog.records:
        assert record.levelname == "WARNING"
        assert (
            "Using 'bunched=False' in this formalism makes the approximation of bunch length = C/(2*pi)."
            in record.message
        )
        assert "Please use the BjorkenMtingwaIBS class for fully accurate results." in record.message
    # --------------------------------------------------------------------
    # Check the growth rates were indeed computed
    assert IBS.ibs_growth_rates is not None
    assert isinstance(IBS.ibs_growth_rates, IBSGrowthRates)
