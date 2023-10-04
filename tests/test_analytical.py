"""
Tests in here are each for a different machine and configuration.
We are computing the growth rates from MAD-X, from the old code
of Michail and from the package's analytical module and ensure
they are consistent.
"""
import numpy as np
import pytest
import xpart as xp

from helpers import get_madx_ibs_beam_size_growth_time

from xibs._old_michail import MichailIBS
from xibs.analytical import BeamParameters, Nagaitsev, OpticsParameters


def test_sps_lhc_ions_growth_rates(matched_sps_lhc_ions_injection):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx = matched_sps_lhc_ions_injection
    madx.command.twiss()  # needs to be called before IBS!
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_beam_size_growth_time(madx)
    # --------------------------------------------------------------------
    # Get the growth rates from the old code (xibs._old_michail)
    pass
    # --------------------------------------------------------------------
    # Get the growth rates from the analytical module (xibs.analytical)
    pass
    # --------------------------------------------------------------------
    # Compare the results and ensure they are consistent


def test_CLIC_DR_growth_rates(madx_CLIC_damping_ring, xsuite_line_CLIC_damping_ring):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx = madx_CLIC_damping_ring
    madx.command.twiss()  # needs to be called before IBS!
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_beam_size_growth_time(madx)
    # --------------------------------------------------------------------
    # Xsuite line for xibs
    line = xsuite_line_CLIC_damping_ring
    # Beam parameters
    bunch_intensity = 4.4e9  # could @pytest.mark.parametrize this
    sigma_z = 1.58e-3
    n_part = int(5e3)
    nemitt_x = 5.6644e-07
    nemitt_y = 3.7033e-09
    # Particles and twiss
    p0 = xp.Particles(mass0=xp.ELECTRON_MASS_EV, q0=1, p0c=2.86e9)
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        particle_ref=p0,
        line=line,
    )
    twiss = line.twiss(particle_ref=p0)
    # Statistical values to compute from
    sig_x = np.std(particles.x[particles.state > 0])
    sig_y = np.std(particles.y[particles.state > 0])
    geom_epsx = (sig_x**2 - (twiss["dx"][0] * sig_delta) ** 2) / twiss["betx"][0]
    geom_epsy = sig_y**2 / twiss["bety"][0]
    sig_delta = np.std(particles.delta[particles.state > 0])
    bunch_length = np.std(particles.zeta[particles.state > 0])
    # --------------------------------------------------------------------
    # Get the growth rates from the old code (xibs._old_michail)
    OLDIBS = MichailIBS()
    OLDIBS.set_beam_parameters(particles)
    OLDIBS.set_optic_functions(twiss)
    OLDIBS.calculate_integrals(geom_epsx, geom_epsy, sig_delta, bunch_length)  # stored internally as .Ixx, .Iyy, .Ipp
    # --------------------------------------------------------------------
    # Get the growth rates from the analytical module (xibs.analytical)
    beamparams = BeamParameters(particles)
    optics = OpticsParameters(twiss)
    NEWIBS = Nagaitsev(beamparams, optics)
    integrals = NEWIBS.integrals(geom_epsx, geom_epsy, sig_delta)
    new_rates = NEWIBS.growth_rates(integrals, bunch_length)
    # --------------------------------------------------------------------
    # Compare the package's results to the ones from Michail's old code
    assert np.isclose(new_rates.Tx, OLDIBS.Ixx)
    assert np.isclose(new_rates.Ty, OLDIBS.Iyy)
    assert np.isclose(new_rates.Tz, OLDIBS.Ipp)
    # Compare the package's results to the MAD-X ones
    assert np.isclose(new_rates.Tx, mad_Tx)
    assert np.isclose(new_rates.Ty, mad_Ty)
    assert np.isclose(new_rates.Tz, mad_Tz)
