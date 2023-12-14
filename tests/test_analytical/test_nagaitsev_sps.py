"""
Tests in here are each for different SPS configurations (inj/top and protons/ions).
We are computing the growth rates from MAD-X, from the old code of Michalis and
from the package's Nagaitsev implementation then compare them all to ensure
they are consistent.

FOR COMPARISONS, PLEASE KEEP IN MIND:
    #  - We are using np.isclose so the reference number is the second argument
    #  - We have to explicitely set atol=0 as the numbers can be very small

In some scenarios, the deviation to MAD-X growth rates is a bit high but as long
as we stick close to the values from Michalis' reference code, we are fine.
"""
import numpy as np
import xtrack as xt

from helpers import get_madx_ibs_growth_rates

from xibs._old_michalis import MichalisIBS
from xibs.analytical import NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters

# ----- Tests at injection energy ----- #


def test_sps_injection_protons_vs_old(madx_sps_injection_protons, xtrack_sps_injection_protons):
    # --------------------------------------------------------------------
    # Get MAD-X setup - only want for the params
    madx, params = madx_sps_injection_protons  # fully set up from the config file
    # --------------------------------------------------------------------
    # Load equivalent xsuite line
    line: xt.Line = xtrack_sps_injection_protons
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters.from_madx(madx)
    beamparams.n_part = madx.sequence.sps.beam.npart  # as line.particle_ref is only 1 particle
    # --------------------------------------------------------------------
    # Get growth rates from the old (xibs._old_michalis) and from new
    # (xibs.analytical.NagaitsevIBS) implementations
    MIBS = MichalisIBS()
    MIBS.set_beam_parameters(line.particle_ref)
    MIBS.set_optic_functions(twiss)
    MIBS.Npart = beamparams.n_part  # also need to update it here - old code is clunky
    MIBS.calculate_integrals(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    IBS = NagaitsevIBS(beamparams, opticsparams)
    IBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    # --------------------------------------------------------------------
    # Now we compare values to old implementation.
    assert np.isclose(IBS.ibs_growth_rates.Tx, MIBS.Ixx, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Ty, MIBS.Iyy, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Tz, MIBS.Ipp, atol=0, rtol=1e-2)


def test_sps_injection_protons_vs_madx(madx_sps_injection_protons):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, params = madx_sps_injection_protons  # fully set up from the config file
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # --------------------------------------------------------------------
    # Load the optics and beam params from MAD-X for maximum consistency
    # (also, we want to use a centered Twiss to get closer to MAD-X results)
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    # --------------------------------------------------------------------
    # Compute the growth rates - method computes integrals by default
    IBS = NagaitsevIBS(beamparams, opticsparams)
    IBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    # --------------------------------------------------------------------
    # Compare the results to MAD-X - deviations are ok as long as we are
    # close to the old code's results (test above)
    assert np.isclose(IBS.ibs_growth_rates.Tx, mad_Tx, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Ty, mad_Ty, atol=0, rtol=20e-2)
    assert np.isclose(IBS.ibs_growth_rates.Tz, mad_Tz, atol=0, rtol=1e-2)


def test_sps_injection_ions_vs_old(madx_sps_injection_ions, xtrack_sps_injection_ions):
    # --------------------------------------------------------------------
    # Get MAD-X setup - only want for the params
    madx, params = madx_sps_injection_ions  # fully set up from the config file
    # --------------------------------------------------------------------
    # Load equivalent xsuite line
    line: xt.Line = xtrack_sps_injection_ions
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters.from_madx(madx)
    beamparams.n_part = madx.sequence.sps.beam.npart  # as line.particle_ref is only 1 particle
    # --------------------------------------------------------------------
    # Get growth rates from the old (xibs._old_michalis) and from new
    # (xibs.analytical.NagaitsevIBS) implementations
    MIBS = MichalisIBS()
    MIBS.set_beam_parameters(line.particle_ref)
    MIBS.set_optic_functions(twiss)
    MIBS.Npart = beamparams.n_part  # also need to update it here - old code is clunky
    MIBS.calculate_integrals(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    IBS = NagaitsevIBS(beamparams, opticsparams)
    IBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    # --------------------------------------------------------------------
    # Now we compare values to old implementation.
    assert np.isclose(IBS.ibs_growth_rates.Tx, MIBS.Ixx, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Ty, MIBS.Iyy, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Tz, MIBS.Ipp, atol=0, rtol=1e-2)


def test_sps_injection_ions_vs_madx(madx_sps_injection_ions):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, params = madx_sps_injection_ions  # fully set up from the config file
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # --------------------------------------------------------------------
    # Load the optics and beam params from MAD-X for maximum consistency
    # (also, we want to use a centered Twiss to get closer to MAD-X results)
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    # --------------------------------------------------------------------
    # Compute the growth rates - method computes integrals by default
    IBS = NagaitsevIBS(beamparams, opticsparams)
    IBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    # --------------------------------------------------------------------
    # Compare the results to MAD-X - deviations are ok as long as we are
    # close to the old code's results (test above)
    assert np.isclose(IBS.ibs_growth_rates.Tx, mad_Tx, atol=0, rtol=45e-2)
    assert np.isclose(IBS.ibs_growth_rates.Ty, mad_Ty, atol=0, rtol=5.5e-2)
    assert np.isclose(IBS.ibs_growth_rates.Tz, mad_Tz, atol=0, rtol=1e-2)


# ----- Tests at top energy ----- #


def test_sps_top_protons_vs_old(madx_sps_top_protons, xtrack_sps_top_protons):
    # --------------------------------------------------------------------
    # Get MAD-X setup - only want for the params
    madx, params = madx_sps_top_protons  # fully set up from the config file
    # --------------------------------------------------------------------
    # Load equivalent xsuite line
    line: xt.Line = xtrack_sps_top_protons
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters.from_madx(madx)
    beamparams.n_part = madx.sequence.sps.beam.npart  # as line.particle_ref is only 1 particle
    # --------------------------------------------------------------------
    # Get growth rates from the old (xibs._old_michalis) and from new
    # (xibs.analytical.NagaitsevIBS) implementations
    MIBS = MichalisIBS()
    MIBS.set_beam_parameters(line.particle_ref)
    MIBS.set_optic_functions(twiss)
    MIBS.Npart = beamparams.n_part  # also need to update it here - old code is clunky
    MIBS.calculate_integrals(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    IBS = NagaitsevIBS(beamparams, opticsparams)
    IBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    # --------------------------------------------------------------------
    # Now we compare values to old implementation.
    assert np.isclose(IBS.ibs_growth_rates.Tx, MIBS.Ixx, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Ty, MIBS.Iyy, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Tz, MIBS.Ipp, atol=0, rtol=1e-2)


def test_sps_top_protons_vs_madx(madx_sps_top_protons):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, params = madx_sps_top_protons  # fully set up from the config file
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # --------------------------------------------------------------------
    # Load the optics and beam params from MAD-X for maximum consistency
    # (also, we want to use a centered Twiss to get closer to MAD-X results)
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    # --------------------------------------------------------------------
    # Compute the growth rates - method computes integrals by default
    IBS = NagaitsevIBS(beamparams, opticsparams)
    IBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    # --------------------------------------------------------------------
    # Compare the results to MAD-X - deviations are ok as long as we are
    # close to the old code's results (test above)
    assert np.isclose(IBS.ibs_growth_rates.Tx, mad_Tx, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Ty, mad_Ty, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Tz, mad_Tz, atol=0, rtol=1e-2)


def test_sps_top_ions_vs_old(madx_sps_top_ions, xtrack_sps_top_ions):
    # --------------------------------------------------------------------
    # Get MAD-X setup - only want for the params
    madx, params = madx_sps_top_ions  # fully set up from the config file
    # --------------------------------------------------------------------
    # Load equivalent xsuite line
    line: xt.Line = xtrack_sps_top_ions
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters.from_madx(madx)
    beamparams.n_part = madx.sequence.sps.beam.npart  # as line.particle_ref is only 1 particle
    # --------------------------------------------------------------------
    # Get growth rates from the old (xibs._old_michalis) and from new
    # (xibs.analytical.NagaitsevIBS) implementations
    MIBS = MichalisIBS()
    MIBS.set_beam_parameters(line.particle_ref)
    MIBS.set_optic_functions(twiss)
    MIBS.Npart = beamparams.n_part  # also need to update it here - old code is clunky
    MIBS.calculate_integrals(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    IBS = NagaitsevIBS(beamparams, opticsparams)
    IBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    # --------------------------------------------------------------------
    # Now we compare values to old implementation.
    assert np.isclose(IBS.ibs_growth_rates.Tx, MIBS.Ixx, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Ty, MIBS.Iyy, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Tz, MIBS.Ipp, atol=0, rtol=1e-2)


def test_sps_top_ions_vs_madx(madx_sps_top_ions):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, params = madx_sps_top_ions  # fully set up from the config file
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # --------------------------------------------------------------------
    # Load the optics and beam params from MAD-X for maximum consistency
    # (also, we want to use a centered Twiss to get closer to MAD-X results)
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    # --------------------------------------------------------------------
    # Compute the growth rates - method computes integrals by default
    IBS = NagaitsevIBS(beamparams, opticsparams)
    IBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    # --------------------------------------------------------------------
    # Compare the results to MAD-X - deviations are ok as long as we are
    # close to the old code's results (test above)
    assert np.isclose(IBS.ibs_growth_rates.Tx, mad_Tx, atol=0, rtol=1e-2)
    assert np.isclose(IBS.ibs_growth_rates.Ty, mad_Ty, atol=0, rtol=1.5e-2)
    assert np.isclose(IBS.ibs_growth_rates.Tz, mad_Tz, atol=0, rtol=1e-2)
