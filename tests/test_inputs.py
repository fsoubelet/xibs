"""
Quick tests for the input classes and their initialization.
"""
import numpy as np

from scipy.constants import c

from xibs.inputs import BeamParameters, OpticsParameters

# ----- BeamParameters ----- #


def test_init_beamparameters(madx_sps_injection_protons, xtrack_sps_injection_protons):
    """Check that the BeamParameters are properly initializing."""
    # --------------------------------------------------------------------
    # Get the MAD-X instance and xtrack Line
    madx, params = madx_sps_injection_protons  # fully set up from the config file
    line = xtrack_sps_injection_protons
    p0 = line.particle_ref
    # --------------------------------------------------------------------
    # Initialize BeamParameters
    beamparams = BeamParameters(p0)
    beamparams.n_part = madx.sequence.sps.beam.npart  # as line.particle_ref is only 1 particle
    # --------------------------------------------------------------------
    # Perform checks
    assert np.isclose(beamparams.n_part, madx.sequence.sps.beam.npart)
    assert np.isclose(beamparams.particle_charge, p0.q0)
    assert np.isclose(beamparams.particle_mass_eV, p0.mass0)
    assert np.isclose(beamparams.total_energy_eV, np.sqrt(p0.p0c[0] ** 2 + p0.mass0**2))
    assert np.isclose(beamparams.gamma_rel, p0.gamma0[0])
    assert np.isclose(beamparams.beta_rel, p0.beta0[0])
    assert np.isclose(beamparams.particle_classical_radius_m, p0.get_classical_particle_radius0())


# ----- BeamParameters constructor classmethods ----- #


def test_init_beamparameters_from_madx(madx_sps_injection_protons, xtrack_sps_injection_protons):
    """Check that the BeamParameters are properly initializing from madx. Compare to a particle_ref."""
    # --------------------------------------------------------------------
    # Get the MAD-X instance and xtrack Line
    madx, _ = madx_sps_injection_protons  # fully set up from the config file
    line = xtrack_sps_injection_protons
    p0 = line.particle_ref
    # --------------------------------------------------------------------
    # Initialize BeamParameters
    from_madx = BeamParameters.from_madx(madx)
    from_part = BeamParameters(p0)
    # --------------------------------------------------------------------
    # Perform checks - the MAD-X and line are equivalent so this should work
    assert np.isclose(from_madx.n_part, madx.sequence.sps.beam.npart)
    assert np.isclose(from_madx.particle_charge, madx.sequence.sps.beam.charge)
    assert np.isclose(from_madx.particle_charge, from_part.particle_charge)
    assert np.isclose(from_madx.particle_mass_eV, madx.sequence.sps.beam.mass * 1e9)  # it is in GeV in MAD-X
    assert np.isclose(from_madx.particle_mass_eV, from_part.particle_mass_eV)
    assert np.isclose(from_madx.total_energy_eV, from_part.total_energy_eV)
    assert np.isclose(from_madx.gamma_rel, madx.sequence.sps.beam.gamma)
    assert np.isclose(from_madx.gamma_rel, from_part.gamma_rel)
    assert np.isclose(from_madx.beta_rel, madx.sequence.sps.beam.beta)
    assert np.isclose(from_madx.beta_rel, from_part.beta_rel)
    assert np.isclose(from_madx.particle_classical_radius_m, from_part.particle_classical_radius_m)


def test_init_beamparameters_from_line(xtrack_sps_injection_protons):
    """Check that the BeamParameters are properly initializing from line. Compare to particle_ref directly."""
    # --------------------------------------------------------------------
    # Get the MAD-X instance and xtrack Line
    line = xtrack_sps_injection_protons
    p0 = line.particle_ref
    # --------------------------------------------------------------------
    # Initialize BeamParameters
    from_line = BeamParameters.from_line(line, n_part=1000)  # use 1000 arbitrarily
    from_part = BeamParameters(p0)
    from_part.n_part = 1000  # use same as above
    # --------------------------------------------------------------------
    # Perform checks - the MAD-X and line are equivalent so this should work
    assert np.isclose(from_line.n_part, from_part.n_part)
    assert np.isclose(from_line.particle_charge, from_part.particle_charge)
    assert np.isclose(from_line.particle_mass_eV, from_part.particle_mass_eV)
    assert np.isclose(from_line.total_energy_eV, from_part.total_energy_eV)
    assert np.isclose(from_line.gamma_rel, from_part.gamma_rel)
    assert np.isclose(from_line.beta_rel, from_part.beta_rel)
    assert np.isclose(from_line.particle_classical_radius_m, from_part.particle_classical_radius_m)


# ----- OpticsParameters ----- #


def test_init_opticsparameters_from_xtrack_twiss(xtrack_sps_injection_protons):
    """Check that the OpticsParameters are properly initializing from xtrack twiss."""
    # --------------------------------------------------------------------
    # Get the xtrack Line and Twiss
    line = xtrack_sps_injection_protons
    twiss = line.twiss(method="4d")
    # --------------------------------------------------------------------
    # Initialize OpticsParameters
    opticsparams = OpticsParameters(twiss)
    # --------------------------------------------------------------------
    # Perform checks
    assert np.allclose(opticsparams.s, twiss.s)
    assert np.isclose(opticsparams.circumference, twiss.s[-1])
    assert np.allclose(opticsparams.betx, twiss.betx)
    assert np.allclose(opticsparams.bety, twiss.bety)
    assert np.allclose(opticsparams.alfx, twiss.alfx)
    assert np.allclose(opticsparams.alfy, twiss.alfy)
    assert np.allclose(opticsparams.dx, twiss.dx)
    assert np.allclose(opticsparams.dy, twiss.dy)
    assert np.allclose(opticsparams.dpx, twiss.dpx)
    assert np.allclose(opticsparams.dpy, twiss.dpy)
    assert np.isclose(opticsparams.slip_factor, twiss["slip_factor"])
    assert np.isclose(opticsparams.revolution_frequency, twiss.beta0 * c / twiss.s[-1])


def test_init_opticsparameters_from_madx_twiss(madx_sps_injection_protons, xtrack_sps_injection_protons):
    """
    Check that the OpticsParameters are properly initializing from MAD-X (cpymad) twiss
    and compare the validity of some values to the one initialized from xtrack twiss.
    """
    # --------------------------------------------------------------------
    # Get the MAD-X instance and xtrack Line then twisses
    madx, params = madx_sps_injection_protons  # fully set up from the config file
    line = xtrack_sps_injection_protons
    mad_twiss = madx.table.twiss.dframe()
    xt_twiss = line.twiss(method="4d")
    # --------------------------------------------------------------------
    # Get some additional necessary parameters from MAD-X
    seq_name = madx.table.twiss.summary.sequence  # works whichever your sequence
    frev_hz = madx.sequence[seq_name].beam.freq0 * 1e6  # beware freq0 is in MHz
    gamma_rel = madx.sequence[seq_name].beam.gamma  # relativistic gamma
    gamma_tr = madx.table.summ.gammatr[0]  # transition gamma
    slipfactor = (1 / (gamma_tr**2)) - (1 / (gamma_rel**2))  # use the xsuite convention!
    # --------------------------------------------------------------------
    # Initialize OpticsParameters in both supported ways
    xt_opticsparams = OpticsParameters(xt_twiss)  # our reference
    mad_opticsparams = OpticsParameters(mad_twiss, slipfactor, frev_hz)
    # --------------------------------------------------------------------
    # Perform checks
    assert np.allclose(mad_opticsparams.s, mad_twiss.s)
    assert np.allclose(mad_opticsparams.betx, mad_twiss.betx)
    assert np.allclose(mad_opticsparams.bety, mad_twiss.bety)
    assert np.allclose(mad_opticsparams.alfx, mad_twiss.alfx)
    assert np.allclose(mad_opticsparams.alfy, mad_twiss.alfy)
    assert np.allclose(mad_opticsparams.dx, mad_twiss.dx)
    assert np.allclose(mad_opticsparams.dy, mad_twiss.dy)
    assert np.allclose(mad_opticsparams.dpx, mad_twiss.dpx)
    assert np.allclose(mad_opticsparams.dpy, mad_twiss.dpy)
    # --------------------------------------------------------------------
    # Check some values against reference OpticsParameters initialized from xtrack twiss
    assert np.isclose(mad_opticsparams.circumference, xt_opticsparams.circumference)
    assert np.isclose(mad_opticsparams.slip_factor, xt_opticsparams.slip_factor, rtol=5e-3)  # within 0.5%
    assert np.isclose(mad_opticsparams.revolution_frequency, xt_opticsparams.revolution_frequency)


# ----- OpticsParameters constructor classmethods ----- #


def test_init_opticsparameters_from_line_constructor(xtrack_sps_injection_protons):
    """Check that the OpticsParameters are properly initializing from xtrack twiss."""
    # --------------------------------------------------------------------
    # Get the xtrack Line, Twiss (for reference) and initialize OpticsParameters
    line = xtrack_sps_injection_protons
    twiss = line.twiss(method="4d")
    opticsparams = OpticsParameters.from_line(line)
    # --------------------------------------------------------------------
    # Perform checks
    assert np.allclose(opticsparams.s, twiss.s)
    assert np.isclose(opticsparams.circumference, twiss.s[-1])
    assert np.allclose(opticsparams.betx, twiss.betx)
    assert np.allclose(opticsparams.bety, twiss.bety)
    assert np.allclose(opticsparams.alfx, twiss.alfx)
    assert np.allclose(opticsparams.alfy, twiss.alfy)
    assert np.allclose(opticsparams.dx, twiss.dx)
    assert np.allclose(opticsparams.dy, twiss.dy)
    assert np.allclose(opticsparams.dpx, twiss.dpx)
    assert np.allclose(opticsparams.dpy, twiss.dpy)
    assert np.isclose(opticsparams.slip_factor, twiss["slip_factor"])
    assert np.isclose(opticsparams.revolution_frequency, twiss.beta0 * c / twiss.s[-1])


def test_init_opticsparameters_from_madx_constructor(
    madx_sps_injection_protons, xtrack_sps_injection_protons
):
    """
    Check that the OpticsParameters are properly initializing from MAD-X (cpymad) twiss
    and compare the validity of some values to the one initialized from xtrack twiss.
    """
    # --------------------------------------------------------------------
    # Get the MAD-X instance and xtrack Line; then twisses (for reference)
    # and then initialize OpticsParameters in both supported ways
    madx, _ = madx_sps_injection_protons  # fully set up from the config file
    line = xtrack_sps_injection_protons
    mad_twiss = madx.table.twiss.dframe()
    xt_twiss = line.twiss(method="4d")
    # --------------------------------------------------------------------
    xt_opticsparams = OpticsParameters(xt_twiss)  # our reference
    mad_opticsparams = OpticsParameters.from_madx(madx)
    # --------------------------------------------------------------------
    # Perform checks
    assert np.allclose(mad_opticsparams.s, mad_twiss.s)
    assert np.allclose(mad_opticsparams.betx, mad_twiss.betx)
    assert np.allclose(mad_opticsparams.bety, mad_twiss.bety)
    assert np.allclose(mad_opticsparams.alfx, mad_twiss.alfx)
    assert np.allclose(mad_opticsparams.alfy, mad_twiss.alfy)
    assert np.allclose(mad_opticsparams.dx, mad_twiss.dx)
    assert np.allclose(mad_opticsparams.dy, mad_twiss.dy)
    assert np.allclose(mad_opticsparams.dpx, mad_twiss.dpx)
    assert np.allclose(mad_opticsparams.dpy, mad_twiss.dpy)
    # --------------------------------------------------------------------
    # Check some values against reference OpticsParameters initialized from xtrack twiss
    assert np.isclose(mad_opticsparams.circumference, xt_opticsparams.circumference)
    assert np.isclose(mad_opticsparams.slip_factor, xt_opticsparams.slip_factor, rtol=5e-3)  # within 0.5%
    assert np.isclose(mad_opticsparams.revolution_frequency, xt_opticsparams.revolution_frequency)
