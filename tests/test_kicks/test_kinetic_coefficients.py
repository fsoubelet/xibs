"""
Tests in here are for different machines.
We are computing the diffusion and friction coefficients and comparing to the ones from old code.

FOR COMPARISONS, PLEASE KEEP IN MIND:
    #  - We are only using the pearson correlation coefficient, which can still be good if there is an offset.
    #  - We are tracking for a small number of turns and particles because otherwise the CI runners will die.
"""
import numpy as np
import xpart as xp
import xtrack as xt

from xibs._old_michalis import MichalisIBS
from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS


def test_kinetic_coefficients_clic_dr(xtrack_clic_damping_ring):
    """Kinetic coefficients in the CLIC DR compared to old code."""
    # --------------------------------------------------------------------
    # Some simple parameters
    bunch_intensity = int(4.5e9)
    n_part = int(1e3)
    sigma_z = 1.58e-3
    nemitt_x = 5.66e-7
    nemitt_y = 3.7e-9
    # --------------------------------------------------------------------
    # Setup line and particles for tracking
    line: xt.Line = xtrack_clic_damping_ring  # already has particle_ref
    for cavity in [element for element in line.elements if isinstance(element, xt.Cavity)]:
        cavity.lag = 180  # we are above transition
    line.build_tracker(extra_headers=["#define XTRACK_MULTIPOLE_NO_SYNRAD"])
    line.optimize_for_tracking()
    twiss = line.twiss()
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        line=line,
    )
    # --------------------------------------------------------------------
    # Create the IBS objects
    beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
    opticsparams = OpticsParameters.from_line(line)
    IBS = KineticKickIBS(beamparams, opticsparams)  # no dy, chooses Nagaitsev
    MIBS = MichalisIBS()
    MIBS.set_beam_parameters(particles)
    MIBS.set_optic_functions(twiss)
    # --------------------------------------------------------------------
    # Computing kick coefficients for both
    IBS.compute_kick_coefficients(particles)
    MIBS.calculate_kinetic_coefficients(particles)
    # --------------------------------------------------------------------
    # Compare the values of diffusion coefficients
    assert np.isclose(IBS.diffusion_coefficients.Dx, MIBS.Dx, atol=0, rtol=1e-3)
    assert np.isclose(IBS.diffusion_coefficients.Dy, MIBS.Dy, atol=0, rtol=1e-3)
    assert np.isclose(IBS.diffusion_coefficients.Dz, MIBS.Dz, atol=0, rtol=1e-3)
    # --------------------------------------------------------------------
    # Compare the values of friction coefficients
    assert np.isclose(IBS.friction_coefficients.Fx, MIBS.Fx, atol=0, rtol=1e-3)
    assert np.isclose(IBS.friction_coefficients.Fy, MIBS.Fy, atol=0, rtol=1e-3)
    assert np.isclose(IBS.friction_coefficients.Fz, MIBS.Fz, atol=0, rtol=1e-3)


def test_kinetic_coefficients_sps_top_ions(xtrack_sps_top_ions):
    """Kinetic coefficients in the SPS, top ions, compared to old code."""
    # --------------------------------------------------------------------
    # Some simple parameters
    bunch_intensity = int(3.5e8)
    n_part = int(1e3)
    sigma_z = 19.7e-2
    nemitt_x = 1.2612e-6
    nemitt_y = 0.9081e-6
    # --------------------------------------------------------------------
    # Setup line and particles for tracking
    line: xt.Line = xtrack_sps_top_ions  # already has particle_ref
    line.build_tracker()
    line.optimize_for_tracking()
    twiss = line.twiss(method="4d")
    # Acceleration parameters
    rf_voltage = 1.7e6  # 1.7MV from the test config
    harmonic_number = 4653
    line["actcse.31632"].lag = 180  # 0 if below transition, 180 if above
    line["actcse.31632"].voltage = rf_voltage  # In Xsuite for ions, do not multiply by charge as in MADX
    line["actcse.31632"].frequency = OpticsParameters.from_line(line).revolution_frequency * harmonic_number
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        line=line,
    )
    # --------------------------------------------------------------------
    # Create the IBS objects
    beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
    opticsparams = OpticsParameters.from_line(line)
    IBS = KineticKickIBS(beamparams, opticsparams)  # no dy, chooses Nagaitsev
    MIBS = MichalisIBS()
    MIBS.set_beam_parameters(particles)
    MIBS.set_optic_functions(twiss)
    # --------------------------------------------------------------------
    # Computing kick coefficients for both
    IBS.compute_kick_coefficients(particles)
    MIBS.calculate_kinetic_coefficients(particles)
    # --------------------------------------------------------------------
    # Compare the values of diffusion coefficients
    assert np.isclose(IBS.diffusion_coefficients.Dx, MIBS.Dx, atol=0, rtol=1e-3)
    assert np.isclose(IBS.diffusion_coefficients.Dy, MIBS.Dy, atol=0, rtol=1e-3)
    assert np.isclose(IBS.diffusion_coefficients.Dz, MIBS.Dz, atol=0, rtol=1e-3)
    # --------------------------------------------------------------------
    # Compare the values of friction coefficients
    assert np.isclose(IBS.friction_coefficients.Fx, MIBS.Fx, atol=0, rtol=1e-3)
    assert np.isclose(IBS.friction_coefficients.Fy, MIBS.Fy, atol=0, rtol=1e-3)
    assert np.isclose(IBS.friction_coefficients.Fz, MIBS.Fz, atol=0, rtol=1e-3)


def test_kinetic_coefficients_sps_top_protons(xtrack_sps_top_protons):
    """Kinetic coefficients in the SPS, top protons, compared to old code."""
    # --------------------------------------------------------------------
    # Some simple parameters
    bunch_intensity = int(1.2e11)
    n_part = int(1e3)
    sigma_z = 9e-2
    nemitt_x = 2.5e-6
    nemitt_y = 2.5e-6
    # --------------------------------------------------------------------
    # Setup line and particles for tracking
    line: xt.Line = xtrack_sps_top_protons  # already has particle_ref
    line.build_tracker()
    line.optimize_for_tracking()
    twiss = line.twiss(method="4d")
    # Acceleration parameters
    rf_voltage = 3e6  # 3MV from the test config
    harmonic_number = 4653
    line["actcse.31632"].lag = 180
    line["actcse.31632"].voltage = rf_voltage
    line["actcse.31632"].frequency = OpticsParameters.from_line(line).revolution_frequency * harmonic_number
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        line=line,
    )
    # --------------------------------------------------------------------
    # Create the IBS objects
    beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
    opticsparams = OpticsParameters.from_line(line)
    IBS = KineticKickIBS(beamparams, opticsparams)  # no dy, chooses Nagaitsev
    MIBS = MichalisIBS()
    MIBS.set_beam_parameters(particles)
    MIBS.set_optic_functions(twiss)
    # --------------------------------------------------------------------
    # Computing kick coefficients for both
    IBS.compute_kick_coefficients(particles)
    MIBS.calculate_kinetic_coefficients(particles)
    # --------------------------------------------------------------------
    # Compare the values of diffusion coefficients
    assert np.isclose(IBS.diffusion_coefficients.Dx, MIBS.Dx, atol=0, rtol=1e-3)
    assert np.isclose(IBS.diffusion_coefficients.Dy, MIBS.Dy, atol=0, rtol=1e-3)
    assert np.isclose(IBS.diffusion_coefficients.Dz, MIBS.Dz, atol=0, rtol=1e-3)
    # --------------------------------------------------------------------
    # Compare the values of friction coefficients
    assert np.isclose(IBS.friction_coefficients.Fx, MIBS.Fx, atol=0, rtol=1e-3)
    assert np.isclose(IBS.friction_coefficients.Fy, MIBS.Fy, atol=0, rtol=1e-3)
    assert np.isclose(IBS.friction_coefficients.Fz, MIBS.Fz, atol=0, rtol=1e-3)


def test_kinetic_coefficients_ps_injection_protons(xtrack_ps_injection_protons):
    """Kinetic coefficients in the PS, top protons, compared to old code."""
    # --------------------------------------------------------------------
    # Some simple parameters
    bunch_intensity = int(8.1e8)
    n_part = int(1e3)
    sigma_z = 56.8e-2
    nemitt_x = 0.8e-6
    nemitt_y = 0.5e-6
    # --------------------------------------------------------------------
    # Setup line and particles for tracking
    line: xt.Line = xtrack_ps_injection_protons  # already has particle_ref
    line.build_tracker()
    line.optimize_for_tracking()
    twiss = line.twiss(method="4d")
    # Acceleration parameters
    rf_voltage = 200e3  # 200kV from the test config
    harmonic_number = 16
    line["pa.c10.11"].lag = 0  # we are below transition
    line["pa.c10.11"].voltage = rf_voltage
    line["pa.c10.11"].frequency = OpticsParameters.from_line(line).revolution_frequency * harmonic_number
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        line=line,
    )
    # --------------------------------------------------------------------
    # Create the IBS objects
    beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
    opticsparams = OpticsParameters.from_line(line)
    IBS = KineticKickIBS(beamparams, opticsparams)  # no dy, chooses Nagaitsev
    MIBS = MichalisIBS()
    MIBS.set_beam_parameters(particles)
    MIBS.set_optic_functions(twiss)
    # --------------------------------------------------------------------
    # Computing kick coefficients for both
    IBS.compute_kick_coefficients(particles)
    MIBS.calculate_kinetic_coefficients(particles)
    # --------------------------------------------------------------------
    # Compare the values of diffusion coefficients
    assert np.isclose(IBS.diffusion_coefficients.Dx, MIBS.Dx, atol=0, rtol=1e-3)
    assert np.isclose(IBS.diffusion_coefficients.Dy, MIBS.Dy, atol=0, rtol=1e-3)
    assert np.isclose(IBS.diffusion_coefficients.Dz, MIBS.Dz, atol=0, rtol=1e-3)
    # --------------------------------------------------------------------
    # Compare the values of friction coefficients
    assert np.isclose(IBS.friction_coefficients.Fx, MIBS.Fx, atol=0, rtol=1e-3)
    assert np.isclose(IBS.friction_coefficients.Fy, MIBS.Fy, atol=0, rtol=1e-3)
    assert np.isclose(IBS.friction_coefficients.Fz, MIBS.Fz, atol=0, rtol=1e-3)


def test_kinetic_coefficients_ps_injection_ions(xtrack_ps_injection_ions):
    """Kinetic coefficients in the PS, top protons, compared to old code."""
    # --------------------------------------------------------------------
    # Some simple parameters
    bunch_intensity = int(8.1e8)
    n_part = int(1e3)
    sigma_z = 0.06987309241848311
    nemitt_x = 0.8e-6
    nemitt_y = 0.5e-6
    # --------------------------------------------------------------------
    # Setup line and particles for tracking
    line: xt.Line = xtrack_ps_injection_ions  # already has particle_ref
    line.build_tracker()
    line.optimize_for_tracking()
    twiss = line.twiss(method="4d")
    # Acceleration parameters
    rf_voltage = 0.0380958e6  # 200kV from the test config
    harmonic_number = 16
    line["pa.c10.11"].lag = 0  # we are below transition
    line["pa.c10.11"].voltage = rf_voltage
    line["pa.c10.11"].frequency = OpticsParameters.from_line(line).revolution_frequency * harmonic_number
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        line=line,
    )
    # --------------------------------------------------------------------
    # Create the IBS objects
    beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
    opticsparams = OpticsParameters.from_line(line)
    IBS = KineticKickIBS(beamparams, opticsparams)  # no dy, chooses Nagaitsev
    MIBS = MichalisIBS()
    MIBS.set_beam_parameters(particles)
    MIBS.set_optic_functions(twiss)
    # --------------------------------------------------------------------
    # Computing kick coefficients for both
    IBS.compute_kick_coefficients(particles)
    MIBS.calculate_kinetic_coefficients(particles)
    # --------------------------------------------------------------------
    # Compare the values of diffusion coefficients
    assert np.isclose(IBS.diffusion_coefficients.Dx, MIBS.Dx, atol=0, rtol=1e-3)
    assert np.isclose(IBS.diffusion_coefficients.Dy, MIBS.Dy, atol=0, rtol=1e-3)
    assert np.isclose(IBS.diffusion_coefficients.Dz, MIBS.Dz, atol=0, rtol=1e-3)
    # --------------------------------------------------------------------
    # Compare the values of friction coefficients
    assert np.isclose(IBS.friction_coefficients.Fx, MIBS.Fx, atol=0, rtol=1e-3)
    assert np.isclose(IBS.friction_coefficients.Fy, MIBS.Fy, atol=0, rtol=1e-3)
    assert np.isclose(IBS.friction_coefficients.Fz, MIBS.Fz, atol=0, rtol=1e-3)


# Maybe one for the LHC too and we're good?