"""
Tests in here are each for a different machine and configuration.
We are computing the growth rates from MAD-X, from the old code
of Michalis and from the package's analytical module and ensure
they are consistent.
"""
import numpy as np
import pytest
import xpart as xp
import xtrack as xt

# from cpymadtools import lhc
from helpers import get_madx_ibs_beam_size_growth_time

from xibs._old_michalis import MichalisIBS
from xibs.analytical import BeamParameters, NagaitsevIBS, OpticsParameters

# def test_sps_lhc_ions_growth_rates(matched_sps_lhc_ions_injection):
#     # --------------------------------------------------------------------
#     # Get the growth rates from MAD-X
#     madx = matched_sps_lhc_ions_injection
#     madx.command.twiss()  # needs to be called before IBS!
#     mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_beam_size_growth_time(madx)
#     # --------------------------------------------------------------------
#     # Get the growth rates from the old code (xibs._old_Michalis)
#     pass
#     # --------------------------------------------------------------------
#     # Get the growth rates from the analytical module (xibs.analytical)
#     pass
#     # --------------------------------------------------------------------
#     # Compare the results and ensure they are consistent


# def test_CLIC_DR_growth_rates(madx_CLIC_damping_ring, xsuite_line_CLIC_damping_ring):
#     # --------------------------------------------------------------------
#     # Get the growth rates from MAD-X
#     madx = madx_CLIC_damping_ring
#     madx.command.twiss()  # needs to be called before IBS!
#     mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_beam_size_growth_time(madx)
#     # --------------------------------------------------------------------
#     # Xsuite line for xibs
#     line = xsuite_line_CLIC_damping_ring
#     # Beam parameters
#     bunch_intensity = 4.4e9  # could @pytest.mark.parametrize this
#     sigma_z = 1.58e-3
#     n_part = int(5e3)
#     nemitt_x = 5.6644e-07
#     nemitt_y = 3.7033e-09
#     # Particles and twiss
#     p0 = xp.Particles(mass0=xp.ELECTRON_MASS_EV, q0=1, p0c=2.86e9)
#     particles = xp.generate_matched_gaussian_bunch(
#         num_particles=n_part,
#         total_intensity_particles=bunch_intensity,
#         nemitt_x=nemitt_x,
#         nemitt_y=nemitt_y,
#         sigma_z=sigma_z,
#         particle_ref=p0,
#         line=line,
#     )
#     twiss = line.twiss(particle_ref=p0)
#     # Statistical values to compute from
#     sig_x = np.std(particles.x[particles.state > 0])
#     sig_y = np.std(particles.y[particles.state > 0])
#     sig_delta = np.std(particles.delta[particles.state > 0])
#     geom_epsx = (sig_x**2 - (twiss["dx"][0] * sig_delta) ** 2) / twiss["betx"][0]
#     geom_epsy = sig_y**2 / twiss["bety"][0]
#     bunch_length = np.std(particles.zeta[particles.state > 0])
#     # --------------------------------------------------------------------
#     # Get the growth rates from the old code (xibs._old_Michalis)
#     OLDIBS = MichalisIBS()
#     OLDIBS.set_beam_parameters(particles)
#     OLDIBS.set_optic_functions(twiss)
#     OLDIBS.calculate_integrals(geom_epsx, geom_epsy, sig_delta, bunch_length)  # stored as .Ixx, .Iyy, .Ipp
#     # --------------------------------------------------------------------
#     # Get the growth rates from the analytical module (xibs.analytical)
#     beamparams = BeamParameters(particles)
#     optics = OpticsParameters(twiss)
#     NEWIBS = NagaitsevIBS(beamparams, optics)
#     integrals = NEWIBS.integrals(geom_epsx, geom_epsy, sig_delta)
#     new_rates = NEWIBS.growth_rates(geom_epsx, geom_epsy, sig_delta, bunch_length)
#     # --------------------------------------------------------------------
#     # Compare the package's results to the ones from Michalis's old code
#     assert np.isclose(new_rates.Tx, float(OLDIBS.Ixx))  # float() to avoid xobjects LinkedArray
#     assert np.isclose(new_rates.Ty, float(OLDIBS.Iyy))  # float() to avoid xobjects LinkedArray
#     assert np.isclose(new_rates.Tz, float(OLDIBS.Ipp))  # float() to avoid xobjects LinkedArray
#     # Compare the package's results to the MAD-X ones
#     # assert np.isclose(new_rates.Tx, mad_Tx)
#     # assert np.isclose(new_rates.Ty, mad_Ty)
#     # assert np.isclose(new_rates.Tz, mad_Tz)


def test_lhc_injection_protons_no_crossing(madx_lhc_injection_protons_no_crossing):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    # madx = madx_lhc_injection_protons_no_crossing  # fully set up from the config file
    # madx.command.twiss()  # needs to be called before IBS! TODO: does it need centre=True?
    # mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_beam_size_growth_time(madx)  # careful about the factor 2 in MAD-X
    # # --------------------------------------------------------------------
    # # Xsuite line, needs th MAKETHIN first
    # lhc.make_lhc_thin(madx, sequence="lhcb1")
    # madx.command.use(sequence="lhcb1")  # need to use the sequence again
    # line = xt.Line.from_madx_sequence(madx.sequence["lhcb1"])
    # p0 = xp.Particles(mass0=madx.beam.mass * 1e9, q0=madx.beam.charge, p0c=madx.beam.pc * 1e9)
    # line.particle_ref = p0
    # twiss = line.twiss(method="4d")
    # TODO: for analytical tests this is not needed, we can just give straight the values
    # of the emittances for the integrals / growth rates calculations and dont need particles
    # --------------------------------------------------------------------
    # Get the growth rates from the old code (xibs._old_Michalis)
    # --------------------------------------------------------------------
    # Get the growth rates from the analytical module (xibs.analytical)
    # --------------------------------------------------------------------
    # Compare the results and ensure they are consistent
    pass
