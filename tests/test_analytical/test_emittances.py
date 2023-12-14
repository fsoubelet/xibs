"""
Tests in here check the emittance evolution calculation for both NagaitsevIBS and BjorkenMtingwaIBS.
The simple and fast case of the PS protons at injection is taken to compute the growth rates used.
"""
import numpy as np
import xtrack as xt

from xibs.analytical import BjorkenMtingwaIBS, IBSGrowthRates, NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters


def test_emittance_evolution(madx_ps_injection_protons, xtrack_ps_injection_protons):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, params = madx_ps_injection_protons  # fully set up from the config file
    # --------------------------------------------------------------------
    # Load the optics, from MAD-X for maximum consistency
    twiss = madx.twiss(centre=True).dframe()
    seq_name = madx.table.twiss.summary.sequence  # works whichever your sequence
    frev_hz = madx.sequence[seq_name].beam.freq0 * 1e6  # beware freq0 is in MHz
    gamma_rel = madx.sequence[seq_name].beam.gamma  # relativistic gamma
    gamma_tr = madx.table.summ.gammatr[0]  # transition gamma
    slipfactor = (1 / (gamma_tr**2)) - (1 / (gamma_rel**2))  # use the xsuite convention!
    opticsparams = OpticsParameters(twiss, slipfactor, frev_hz)
    # --------------------------------------------------------------------
    # Load the beam parameters from the equivalent xsuite line
    line: xt.Line = xtrack_ps_injection_protons
    beamparams = BeamParameters(line.particle_ref)
    beamparams.n_part = madx.sequence[seq_name].beam.npart  # as line.particle_ref is only 1 particle
    # --------------------------------------------------------------------
    # Set hardcoded growth rates - faster test this way - we don't care about being
    # realistic here, just that with these specific values the computation is correct
    rates = IBSGrowthRates(Tx=-8.534561002903093e-05, Ty=2.444062101488359e-05, Tz=0.00033520513993030626)
    NIBS = NagaitsevIBS(beamparams, opticsparams)
    NIBS.ibs_growth_rates = rates
    BMIBS = BjorkenMtingwaIBS(beamparams, opticsparams)
    BMIBS.ibs_growth_rates = rates
    # --------------------------------------------------------------------
    # Compute new emittances without specifying the time step - it will fallback to revolution frequency
    # We compare to expected values, for NagaitsevIBS and BjorkenMtingwaIBS
    epsx, epsy, sigd, bl = NIBS.emittance_evolution(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length
    )
    assert np.isclose(epsx, 2.685585551672946e-07, atol=0)
    assert np.isclose(epsy, 1.6784909702029845e-07, atol=0)
    assert np.isclose(sigd, 0.0002740293585944697, atol=0)
    assert np.isclose(bl, 0.5684116449445993, atol=0)
    epsx, epsy, sigd, bl = BMIBS.emittance_evolution(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length
    )
    assert np.isclose(epsx, 2.685585551672946e-07, atol=0)
    assert np.isclose(epsy, 1.6784909702029845e-07, atol=0)
    assert np.isclose(sigd, 0.0002740293585944697, atol=0)
    assert np.isclose(bl, 0.5684116449445993, atol=0)
    # --------------------------------------------------------------------
    # Compute new emittances this time when specifying the time step (arbitrary number is fine)
    # We compare to expected values, for NagaitsevIBS and BjorkenMtingwaIBS
    epsx, epsy, sigd, bl = NIBS.emittance_evolution(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length, dt=1
    )
    assert np.isclose(epsx, 2.685356359022883e-07, atol=0)
    assert np.isclose(epsy, 1.6785319939752902e-07, atol=0)
    assert np.isclose(sigd, 0.0002740752903667045, atol=0)
    assert np.isclose(bl, 0.5685069199704035, atol=0)
    epsx, epsy, sigd, bl = BMIBS.emittance_evolution(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length, dt=1
    )
    assert np.isclose(epsx, 2.685356359022883e-07, atol=0)
    assert np.isclose(epsy, 1.6785319939752902e-07, atol=0)
    assert np.isclose(sigd, 0.0002740752903667045, atol=0)
    assert np.isclose(bl, 0.5685069199704035, atol=0)
