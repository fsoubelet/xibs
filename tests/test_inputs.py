"""
Quick tests for the input classes and their initialization.
"""
import numpy as np

from scipy.constants import c

from xibs.inputs import BeamParameters, OpticsParameters


def test_init_beamparameters(madx_lhc_injection_protons, xtrack_lhc_injection_protons):
    """Check that the BeamParameters are properly initializing."""
    # --------------------------------------------------------------------
    # Get the MAD-X instance and xtrack Line
    madx, params = madx_lhc_injection_protons  # fully set up from the config file
    line = xtrack_lhc_injection_protons
    p0 = line.particle_ref
    # --------------------------------------------------------------------
    # Initialize BeamParameters
    beamparams = BeamParameters(p0)
    beamparams.n_part = madx.sequence.lhcb1.beam.npart  # as line.particle_ref is only 1 particle
    # --------------------------------------------------------------------
    # Perform checks
    assert np.isclose(beamparams.n_part, madx.sequence.lhcb1.beam.npart)
    assert np.isclose(beamparams.particle_charge, p0.q0)
    assert np.isclose(beamparams.particle_mass_GeV, p0.mass0 * 1e-9)
    assert np.isclose(beamparams.total_energy_GeV, np.sqrt(p0.p0c[0] ** 2 + p0.mass0**2) * 1e-9)
    assert np.isclose(beamparams.gamma_rel, p0.gamma0[0])
    assert np.isclose(beamparams.beta_rel, p0.beta0[0])
    assert np.isclose(beamparams.particle_classical_radius_m, p0.get_classical_particle_radius0())


def test_init_opticsparameters_from_xtrack_twiss(xtrack_lhc_injection_protons):
    """Check that the OpticsParameters are properly initializing from xtrack twiss."""
    # --------------------------------------------------------------------
    # Get the xtrack Line and Twiss
    line = xtrack_lhc_injection_protons
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


def test_init_opticsparameters_from_madx_twiss(madx_lhc_injection_protons, xtrack_lhc_injection_protons):
    """
    Check that the OpticsParameters are properly initializing from MAD-X (cpymad) twiss
    and compare the validity of some values to the one initialized from xtrack twiss.
    """
    # --------------------------------------------------------------------
    # Get the MAD-X instance and xtrack Line then twisses
    madx, params = madx_lhc_injection_protons  # fully set up from the config file
    line = xtrack_lhc_injection_protons
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
    assert np.isclose(mad_opticsparams.slip_factor, xt_opticsparams.slip_factor, rtol=1e-3)  # within 0.1%
    assert np.isclose(mad_opticsparams.revolution_frequency, xt_opticsparams.revolution_frequency)
