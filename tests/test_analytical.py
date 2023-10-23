"""
Tests in here are each for a different machine and configuration.
We are computing the growth rates from MAD-X, from the old code
of Michalis and from the package's analytical module and ensure
they are consistent.
"""
import numpy as np
import xpart as xp
import xtrack as xt

# from cpymadtools import lhc
from helpers import get_madx_ibs_growth_rates

from xibs._old_michalis import MichalisIBS
from xibs.analytical import BeamParameters, NagaitsevIBS, OpticsParameters


def test_lhc_injection_protons_no_crossing(madx_lhc_injection_protons_no_crossing):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    # madx = madx_lhc_injection_protons_no_crossing  # fully set up from the config file
    # mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)  # careful about the factor 2 in MAD-X
    # --------------------------------------------------------------------
    # Load xsuite line
    # TODO: get from a fixture
    # p0 = xp.Particles(mass0=madx.beam.mass * 1e9, q0=madx.beam.charge, p0c=madx.beam.pc * 1e9)
    # line.particle_ref = p0
    # twiss = line.twiss(method="4d")
    # TODO: for analytical tests this is not needed, we can just give straight the values
    # of the geometrical emittances for the integrals / growth rates calculations and dont
    # need xpart.Particles object. We can get them from the madx instance's beam.
    # --------------------------------------------------------------------
    # Get the growth rates from the old code (xibs._old_Michalis)
    # --------------------------------------------------------------------
    # Get the growth rates from the analytical module (xibs.analytical)
    # --------------------------------------------------------------------
    # Compare the results and ensure they are consistent
    pass
