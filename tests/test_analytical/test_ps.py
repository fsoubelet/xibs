"""
Tests in here are each for different SPS configurations (inj/top and protons/ions).
We are computing the growth rates from MAD-X, from the old code of Michalis and
from the package's (new & fast) analytical module then compare them all to ensure
they are consistent.
"""
import numpy as np
import xpart as xp
import xtrack as xt

from helpers import get_madx_ibs_growth_rates

from xibs._old_michalis import MichalisIBS
from xibs.analytical import NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters


def test_ps_injection_protons(madx_ps_injection_protons, xtrack_ps_injection_protons):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, params = madx_ps_injection_protons  # fully set up from the config file
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # --------------------------------------------------------------------
    # Load equivalent xsuite line
    line = xtrack_ps_injection_protons
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters(line.particle_ref)
    beamparams.n_part = madx.sequence.ps.beam.npart  # as line.particle_ref is only 1 particle
    # --------------------------------------------------------------------
    # Get the growth rates from the old code (xibs._old_michalis)
    MIBS = MichalisIBS()
    MIBS.set_beam_parameters(line.particle_ref)
    MIBS.set_optic_functions(twiss)
    MIBS.Npart = beamparams.n_part  # also need to update it here
    MIBS.calculate_integrals(
        Emit_x=params.geom_epsx, Emit_y=params.geom_epsy, Sig_M=params.sig_delta, BunchL=params.bunch_length
    )
    # --------------------------------------------------------------------
    # Get the growth rates from the analytical module (xibs.analytical)
    IBS = NagaitsevIBS(beamparams, opticsparams)
    IBS.integrals(params.geom_epsx, params.geom_epsy, params.sig_delta)
    IBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    # --------------------------------------------------------------------
    # Now we compare values. PLEASE KEEP IN MIND:
    #  - We are using np.isclose so the reference number is the second argument
    #  - We have to explicitely set atol=0 as the numbers are very small
    #  - We leave the default of rtol=1e-5 so we compare within 0.001%
    # Compare the results to old implementation
    np.isclose(IBS.ibs_growth_rates.Tx, MIBS.Ixx, atol=0)
    np.isclose(IBS.ibs_growth_rates.Ty, MIBS.Iyy, atol=0)
    np.isclose(IBS.ibs_growth_rates.Tz, MIBS.Ipp, atol=0)
    # Compare the results to MAD-X (allow 10% deviation as MAD-X takes vertical disp into account)
    np.isclose(IBS.ibs_growth_rates.Tx, mad_Tx, atol=0)
    np.isclose(IBS.ibs_growth_rates.Ty, mad_Ty, atol=0)
    np.isclose(IBS.ibs_growth_rates.Tz, mad_Tz, atol=0)


def test_ps_injection_ions(madx_ps_injection_ions, xtrack_ps_injection_ions):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, params = madx_ps_injection_ions  # fully set up from the config file
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # --------------------------------------------------------------------
    # Load equivalent xsuite line
    line = xtrack_ps_injection_ions
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters(line.particle_ref)
    beamparams.n_part = madx.sequence.ps.beam.npart  # as line.particle_ref is only 1 particle
    # --------------------------------------------------------------------
    # Get the growth rates from the old code (xibs._old_michalis)
    MIBS = MichalisIBS()
    MIBS.set_beam_parameters(line.particle_ref)
    MIBS.set_optic_functions(twiss)
    MIBS.Npart = beamparams.n_part  # also need to update it here
    MIBS.calculate_integrals(
        Emit_x=params.geom_epsx, Emit_y=params.geom_epsy, Sig_M=params.sig_delta, BunchL=params.bunch_length
    )
    # --------------------------------------------------------------------
    # Get the growth rates from the analytical module (xibs.analytical)
    IBS = NagaitsevIBS(beamparams, opticsparams)
    IBS.integrals(params.geom_epsx, params.geom_epsy, params.sig_delta)
    IBS.growth_rates(params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length)
    # --------------------------------------------------------------------
    # Now we compare values. PLEASE KEEP IN MIND:
    #  - We are using np.isclose so the reference number is the second argument
    #  - We have to explicitely set atol=0 as the numbers are very small
    #  - We leave the default of rtol=1e-5 so we compare within 0.001%
    # Compare the results to old implementation
    np.isclose(IBS.ibs_growth_rates.Tx, MIBS.Ixx, atol=0)
    np.isclose(IBS.ibs_growth_rates.Ty, MIBS.Iyy, atol=0)
    np.isclose(IBS.ibs_growth_rates.Tz, MIBS.Ipp, atol=0)
    # Compare the results to MAD-X (allow 10% deviation as MAD-X takes vertical disp into account)
    np.isclose(IBS.ibs_growth_rates.Tx, mad_Tx, atol=0)
    np.isclose(IBS.ibs_growth_rates.Ty, mad_Ty, atol=0)
    np.isclose(IBS.ibs_growth_rates.Tz, mad_Tz, atol=0)
