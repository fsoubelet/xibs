"""
Tests in here check that some specific behaviours are implemented correctly in
the kick classes for some edge cases.

For example, check the initialization of SimpleKickIBS based on the detected vertical dispersion
in the machine.

We also check for instance that the BjorkenMtingwaIBS issues a warning when it is
asked to compute IBS growth rates with a Twiss that was not centered.

We also check that providing (equivalent) geometric or normalized emittances does
not change the results of the calculations.
"""
import logging

import xpart as xp
import xtrack as xt

from xibs.analytical import BjorkenMtingwaIBS, NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS, SimpleKickIBS


def test_simple_kick_chooses_bjorken_mtingwa_with_vertical_dispersion(
    madx_lhc_injection_protons_with_vertical_disp, caplog
):
    """
    Checking that SimpleKickIBS initialization logs a message about its choice of analytical
    formalism and makes the correct one when no vertical dispersion is in the machine, aka
    using NagaitsevIBS. Here we use LHC injection protons with xing angles MAD-X as it has
    vertical dispersion.
    """
    caplog.set_level(logging.ERROR)  # so we don't catch warnings from input parameters
    # --------------------------------------------------------------------
    # Get the inputs from MAD-X and initialize IBS class
    madx, params = madx_lhc_injection_protons_with_vertical_disp  # fully set up from the config file
    opticsparams = OpticsParameters.from_madx(madx)  # logs warning because of betatron coupling
    beamparams = BeamParameters.from_madx(madx)
    caplog.set_level(logging.INFO)  # now let's capture the interesting messages
    IBS = SimpleKickIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Check that the chosen analytical formalism is Nagaitsev
    assert isinstance(IBS.analytical_ibs, BjorkenMtingwaIBS)
    # Check proper linking betweeb instance attributes
    assert IBS.beam_parameters is IBS.analytical_ibs.beam_parameters
    assert IBS.optics is IBS.analytical_ibs.optics
    # --------------------------------------------------------------------
    # Check that the choice has been logged to the user, with the possibility
    # of overriding it
    record = caplog.records[0]  # formalism choice info should be in the first logged message
    assert record.levelname == "INFO"
    assert (
        "Non-zero vertical dispersion detected in the lattice, using Bjorken & Mtingwa formalism"
        in record.message
    )
    record = caplog.records[-1]  # overriding possibility should be in the last logged message
    assert record.levelname == "INFO"
    assert (
        "This can be overridden manually, by explicitely setting the self.analytical_ibs attribute"
        in record.message
    )


def test_simple_kick_chooses_nagaitsev_without_vertical_dispersion(madx_sps_injection_protons, caplog):
    """
    Checking that SimpleKickIBS initialization logs a message about its choice of analytical
    formalism and makes the correct one when no vertical dispersion is in the machine, aka
    using NagaitsevIBS. We use SPS injection protons MAD-X as it has no vertical dispersion.
    """
    caplog.set_level(logging.ERROR)  # so we don't catch warnings from input parameters
    # --------------------------------------------------------------------
    # Get the inputs from MAD-X and initialize IBS class
    madx, params = madx_sps_injection_protons  # fully set up from the config file
    opticsparams = OpticsParameters.from_madx(madx)
    beamparams = BeamParameters.from_madx(madx)
    caplog.set_level(logging.INFO)  # now let's capture the interesting messages
    IBS = SimpleKickIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Check that the chosen analytical formalism is Nagaitsev
    assert isinstance(IBS.analytical_ibs, NagaitsevIBS)
    # Check proper linking betweeb instance attributes
    assert IBS.beam_parameters is IBS.analytical_ibs.beam_parameters
    assert IBS.optics is IBS.analytical_ibs.optics
    # --------------------------------------------------------------------
    # Check that the choice has been logged to the user, with the possibility
    # of overriding it
    record = caplog.records[0]  # formalism choice info should be in the first logged message
    assert record.levelname == "INFO"
    assert "No vertical dispersion in the lattice, using Nagaitsev formalism" in record.message
    record = caplog.records[-1]  # overriding possibility should be in the last logged message
    assert record.levelname == "INFO"
    assert (
        "This can be overridden manually, by explicitely setting the self.analytical_ibs attribute"
        in record.message
    )


def test_simple_kick_propagates_when_overwriting_analytical_ibs(madx_sps_injection_protons, caplog):
    """
    Checking that SimpleKickIBS logs and propagates when the user manually overwrites the
    analytical_ibs attribute with its own.
    """
    caplog.set_level(logging.ERROR)  # so we don't catch warnings from input parameters
    # --------------------------------------------------------------------
    # Get the inputs from MAD-X and initialize IBS class
    madx, params = madx_sps_injection_protons  # fully set up from the config file
    opticsparams = OpticsParameters.from_madx(madx)  # logs warning because of betatron coupling
    beamparams = BeamParameters.from_madx(madx)
    IBS = SimpleKickIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Check that the chosen analytical formalism is Nagaitsev
    assert isinstance(IBS.analytical_ibs, NagaitsevIBS)
    # Check proper linking betweeb instance attributes
    assert IBS.beam_parameters is IBS.analytical_ibs.beam_parameters
    assert IBS.optics is IBS.analytical_ibs.optics
    # --------------------------------------------------------------------
    # Overwrite with a BjorkenMtingwaIBS instance and check
    newbeamparams = BeamParameters.from_madx(madx)
    newbeamparams.npart = 50  # easy to check
    newopticsparams = OpticsParameters.from_madx(madx)
    newopticsparams.circumference = 100  # easy to check
    newanalytical = BjorkenMtingwaIBS(newbeamparams, newopticsparams)
    caplog.set_level(logging.DEBUG)  # now let's capture the interesting messages
    IBS.analytical_ibs = newanalytical
    assert isinstance(IBS.analytical_ibs, BjorkenMtingwaIBS)
    assert IBS.beam_parameters.npart == 50
    assert IBS.optics.circumference == 100
    # --------------------------------------------------------------------
    # Check that the choice has been logged to the user, with the possibility
    # of overriding it
    record = caplog.records[0]  # overwriting info in first message
    assert record.levelname == "DEBUG"
    assert "Overwriting the analytical ibs implementation used for growth rates calculation" in record.message
    record = caplog.records[-1]  # propagation info in second (here also last) message
    assert record.levelname == "DEBUG"
    assert (
        "Re-pointing the instance's beam and optics parameters to that of the new analytical implementation"
        in record.message
    )


def test_simple_kick_auto_recomputes_rates(xtrack_sps_top_ions):
    """
    Test with Pb ions in the SPS that the auto-recomputing of IBS kick coefficients works as intended.
    For this test we will use fake values for the beam parameters to be in a regime that 'stimulates'
    IBS, with few particles, and we will
    """
    # --------------------------------------------------------------------
    # Some exaggerated parameters
    bunch_intensity = int(5e11)
    n_part = int(200)
    sigma_z = 5e-2
    nemitt_x = 1.0e-6
    nemitt_y = 0.25e-6
    rf_voltage = 1.7e6  # 1.7MV
    harmonic_number = 4653
    # --------------------------------------------------------------------
    # Setup line and particles
    line: xt.Line = xtrack_sps_top_ions  # already has particle_ref
    line["actcse.31632"].lag = 180  # above transition
    line["actcse.31632"].voltage = rf_voltage  # from config
    line["actcse.31632"].frequency = OpticsParameters.from_line(line).revolution_frequency * harmonic_number
    line.build_tracker()
    line.optimize_for_tracking()
    line.twiss(method="4d")
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        line=line,
    )
    # --------------------------------------------------------------------
    # Create the IBS parameters and class
    PERCENT = 0.05  # trigger recompute if an emittance changes by more than PERCENT%
    beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
    opticsparams = OpticsParameters.from_line(line)
    IBS = SimpleKickIBS(beamparams, opticsparams, auto_recompute_coefficients_percent=PERCENT)
    initial_coefficients = IBS.compute_kick_coefficients(particles)
    # --------------------------------------------------------------------
    # Some initial asserts
    assert IBS.auto_recompute_coefficients_percent == PERCENT
    assert IBS._need_to_recompute_coefficients is False
    # --------------------------------------------------------------------
    # Do one kick application and it should set the recompute flag for the following one
    # because with our beam conditions at least 1 emittance will have changed by PERCENT%
    IBS.apply_ibs_kick(particles)
    assert IBS._need_to_recompute_coefficients is True
    # --------------------------------------------------------------------
    # Track one turn and apply kick, and check the rates have been recomputed (and changed)
    line.track(particles, num_turns=1)
    IBS.apply_ibs_kick(particles)
    assert IBS.kick_coefficients != initial_coefficients


def test_kinetic_kick_auto_recomputes_rates(xtrack_sps_top_ions):
    """
    Test with Pb ions in the SPS that the auto-recomputing of IBS kick coefficients works as intended.
    For this test we will use fake values for the beam parameters to be in a regime that 'stimulates'
    IBS, with few particles, and we will
    """
    # --------------------------------------------------------------------
    # Some exaggerated parameters
    bunch_intensity = int(5e11)
    n_part = int(200)
    sigma_z = 5e-2
    nemitt_x = 1.0e-6
    nemitt_y = 0.25e-6
    rf_voltage = 1.7e6  # 1.7MV
    harmonic_number = 4653
    # --------------------------------------------------------------------
    # Setup line and particles
    line: xt.Line = xtrack_sps_top_ions  # already has particle_ref
    line["actcse.31632"].lag = 180  # above transition
    line["actcse.31632"].voltage = rf_voltage  # from config
    line["actcse.31632"].frequency = OpticsParameters.from_line(line).revolution_frequency * harmonic_number
    line.build_tracker()
    line.optimize_for_tracking()
    line.twiss(method="4d")
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        line=line,
    )
    # --------------------------------------------------------------------
    # Create the IBS parameters and class
    PERCENT = 0.05  # trigger recompute if an emittance changes by more than PERCENT%
    beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
    opticsparams = OpticsParameters.from_line(line)
    IBS = KineticKickIBS(beamparams, opticsparams, auto_recompute_coefficients_percent=PERCENT)
    initial_coefficients = IBS.compute_kick_coefficients(particles)
    # --------------------------------------------------------------------
    # Some initial asserts
    assert IBS.auto_recompute_coefficients_percent == PERCENT
    assert IBS._need_to_recompute_coefficients is False
    # --------------------------------------------------------------------
    # Do one kick application and it should set the recompute flag for the following one
    # because with our beam conditions at least 1 emittance will have changed by PERCENT%
    IBS.apply_ibs_kick(particles)
    assert IBS._need_to_recompute_coefficients is True
    # --------------------------------------------------------------------
    # Track one turn and apply kick, and check the rates have been recomputed (and changed)
    line.track(particles, num_turns=1)
    IBS.apply_ibs_kick(particles)
    assert IBS.kick_coefficients != initial_coefficients
