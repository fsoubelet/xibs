"""
Tests in here are for a few different machines, all above transition energy.
We are getting emittances with IBS kicks in tracking, and comparing to the analytical evolutions.

FOR COMPARISONS, PLEASE KEEP IN MIND:
    #  - We are only using the pearson correlation coefficient, which can still be good if there is an offset.
    #  - We are tracking for a small number of turns and particles because otherwise the CI runners will die.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xpart as xp
import xtrack as xt

from helpers import Records
from scipy.stats import pearsonr

from xibs.analytical import NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import IBSKickCoefficients, SimpleKickIBS


def test_simple_kicks_lead_to_increased_momenta(xtrack_sps_top_ions):
    """
    Do an IBS kick on Pb ions in the SPS and make sure the momenta have increased.
    always be the case in this formalism. Using fake high values for the beam parameters
    in order to be in a regime that 'stimulates' IBS.
    """
    # --------------------------------------------------------------------
    # Some simple parameters
    bunch_intensity = int(3.5e11)
    n_part = int(1e3)  # we don't want too many particles for CI
    sigma_z = 8e-2
    nemitt_x = 1.0e-6
    nemitt_y = 0.25e-6
    harmonic_number = 4653
    # --------------------------------------------------------------------
    # Setup line and particles for tracking
    line: xt.Line = xtrack_sps_top_ions  # already has particle_ref
    line["actcse.31632"].lag = 180  # above transition
    line["actcse.31632"].voltage = 1.7e6  # from config
    line["actcse.31632"].frequency = OpticsParameters.from_line(line).revolution_frequency * harmonic_number
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        line=line,
    )
    particles2 = particles.copy()
    # --------------------------------------------------------------------
    # Create IBS object, put high kick coefficients and apply IBS kicks
    beamparams = BeamParameters.from_line(line, n_part=bunch_intensity)
    opticsparams = OpticsParameters.from_line(line)
    IBS = SimpleKickIBS(beamparams, opticsparams)  # no dy, chooses Nagaitsev
    IBS.kick_coefficients = IBSKickCoefficients(1e-4, 1e-4, 1e-4)
    IBS.apply_ibs_kick(particles)
    # --------------------------------------------------------------------
    # Compare to initial distribution, make sure momenta have increased
    assert np.std(particles.px) > np.std(particles2.px)
    assert np.std(particles.py) > np.std(particles2.py)
    assert np.std(particles.delta) > np.std(particles2.delta)


def test_simple_kicks_clic_dr(xtrack_clic_damping_ring):
    """Track positrons in the CLIC DR and compare to analytical."""
    # --------------------------------------------------------------------
    # Some simple parameters
    bunch_intensity = int(4.5e9)
    n_part = int(1e3)  # we don't want too many particles for CI
    sigma_z = 1.58e-3
    nemitt_x = 5.66e-7
    nemitt_y = 3.7e-9
    nturns = 500  # number of turns to loop for
    ibs_step = 50  # frequency to re-compute the growth rates & kick coefficients in [turns]
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
    IBS = SimpleKickIBS(beamparams, opticsparams)  # no dy, chooses Nagaitsev
    NIBS = NagaitsevIBS(beamparams, opticsparams)
    # --------------------------------------------------------------------
    # Prepare records for data storage & store the initial values
    kicked_tbt = Records.init_zeroes(nturns)
    kicked_tbt.update_at_turn(0, particles, twiss)
    analytical_tbt = Records.init_zeroes(nturns)
    analytical_tbt.update_at_turn(0, particles, twiss)
    # --------------------------------------------------------------------
    # Do the tracking, with IBS kicks
    for turn in range(1, nturns):
        # ----- Potentially re-compute the IBS growth rates and kick coefficients ----- #
        if (turn % ibs_step == 0) or (turn == 1):
            # Compute kick coefficients from the particle distribution at this moment
            IBS.compute_kick_coefficients(particles)
            # Compute analytical values from those at the previous turn
            NIBS.growth_rates(
                analytical_tbt.epsilon_x[turn - 1],
                analytical_tbt.epsilon_y[turn - 1],
                analytical_tbt.sigma_delta[turn - 1],
                analytical_tbt.bunch_length[turn - 1],
            )
        # ----- Manually Apply IBS Kick, Track Turn & get values ----- #
        IBS.apply_ibs_kick(particles)
        line.track(particles, num_turns=1)
        kicked_tbt.update_at_turn(turn, particles, twiss)
        ana_emit_x, ana_emit_y, ana_sig_delta, ana_bunch_length = NIBS.emittance_evolution(
            analytical_tbt.epsilon_x[turn - 1],
            analytical_tbt.epsilon_y[turn - 1],
            analytical_tbt.sigma_delta[turn - 1],
            analytical_tbt.bunch_length[turn - 1],
        )
        analytical_tbt.epsilon_x[turn] = ana_emit_x
        analytical_tbt.epsilon_y[turn] = ana_emit_y
        analytical_tbt.sigma_delta[turn] = ana_sig_delta
        analytical_tbt.bunch_length[turn] = ana_bunch_length
    # --------------------------------------------------------------------
    # Eventually plotting code to upload artifacts and check visually should we want
    plot_kicks_vs_analytical(kicked_tbt, analytical_tbt, "simple_kicks_clic_dr")
    # --------------------------------------------------------------------
    # Do some checks - we want some level of positive correlation between kicks and analytical
    assert pearsonr(kicked_tbt.epsilon_x, analytical_tbt.epsilon_x).statistic > 0
    assert pearsonr(kicked_tbt.epsilon_y, analytical_tbt.epsilon_y).statistic > 0
    assert pearsonr(kicked_tbt.sigma_delta, analytical_tbt.sigma_delta).statistic > 0
    assert pearsonr(kicked_tbt.bunch_length, analytical_tbt.bunch_length).statistic > 0


def test_simple_kicks_sps_top_ions(xtrack_sps_top_ions):
    """
    Track Pb ions in the SPS and compare to analytical. For this test we will use fake values
    for the beam parameters in order to be in a regime that 'stimulates' IBS. This way we have
    sigmificant emittance growth and not oscillations around a stable point.
    """
    # --------------------------------------------------------------------
    # Some simple parameters
    bunch_intensity = int(3.5e11)
    n_part = int(1e3)  # we don't want too many particles for CI
    sigma_z = 8e-2
    nemitt_x = 1.0e-6
    nemitt_y = 0.25e-6
    harmonic_number = 4653
    nturns = 1000  # number of turns to loop for
    ibs_step = 50  # frequency to re-compute the growth rates & kick coefficients in [turns]
    # --------------------------------------------------------------------
    # Setup line and particles for tracking
    line: xt.Line = xtrack_sps_top_ions  # already has particle_ref
    line["actcse.31632"].lag = 180  # above transition
    line["actcse.31632"].voltage = 1.7e6  # from config
    line["actcse.31632"].frequency = OpticsParameters.from_line(line).revolution_frequency * harmonic_number
    line.build_tracker()
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
    IBS = SimpleKickIBS(beamparams, opticsparams)  # no dy, chooses Nagaitsev
    NIBS = NagaitsevIBS(beamparams, opticsparams)  # same analytical to compare
    # --------------------------------------------------------------------
    # Prepare records for data storage & store the initial values
    kicked_tbt = Records.init_zeroes(nturns)
    kicked_tbt.update_at_turn(0, particles, twiss)
    analytical_tbt = Records.init_zeroes(nturns)
    analytical_tbt.update_at_turn(0, particles, twiss)
    # --------------------------------------------------------------------
    # Do the tracking, with IBS kicks
    for turn in range(1, nturns):
        # ----- Potentially re-compute the IBS growth rates and kick coefficients ----- #
        if (turn % ibs_step == 0) or (turn == 1):
            # Compute kick coefficients from the particle distribution at this moment
            IBS.compute_kick_coefficients(particles)
            # Compute analytical values from those at the previous turn
            NIBS.growth_rates(
                analytical_tbt.epsilon_x[turn - 1],
                analytical_tbt.epsilon_y[turn - 1],
                analytical_tbt.sigma_delta[turn - 1],
                analytical_tbt.bunch_length[turn - 1],
            )
        # ----- Manually Apply IBS Kick, Track Turn & get values ----- #
        IBS.apply_ibs_kick(particles)
        line.track(particles, num_turns=1)
        kicked_tbt.update_at_turn(turn, particles, twiss)
        ana_emit_x, ana_emit_y, ana_sig_delta, ana_bunch_length = NIBS.emittance_evolution(
            analytical_tbt.epsilon_x[turn - 1],
            analytical_tbt.epsilon_y[turn - 1],
            analytical_tbt.sigma_delta[turn - 1],
            analytical_tbt.bunch_length[turn - 1],
        )
        analytical_tbt.epsilon_x[turn] = ana_emit_x
        analytical_tbt.epsilon_y[turn] = ana_emit_y
        analytical_tbt.sigma_delta[turn] = ana_sig_delta
        analytical_tbt.bunch_length[turn] = ana_bunch_length
    # --------------------------------------------------------------------
    # Eventually plotting code to upload artifacts and check visually should we want
    plot_kicks_vs_analytical(kicked_tbt, analytical_tbt, "simple_kicks_sps_top_ions")
    # --------------------------------------------------------------------
    # Do some checks - we want some level of positive correlation between kicks and analytical
    assert pearsonr(kicked_tbt.epsilon_x, analytical_tbt.epsilon_x).statistic > 0
    # Vertical is very stagnant in this scenario so we forgo this check as pearson can't handle it
    # assert pearsonr(kicked_tbt.epsilon_y, analytical_tbt.epsilon_y).statistic > 0
    assert pearsonr(kicked_tbt.sigma_delta, analytical_tbt.sigma_delta).statistic > 0
    assert pearsonr(kicked_tbt.bunch_length, analytical_tbt.bunch_length).statistic > 0


# ----- Plotting Helper ----- #


def plot_kicks_vs_analytical(kicktbt: Records, analyticaltbt: Records, filename: str):
    nturns = len(kicktbt.epsilon_x)
    turns = np.arange(nturns, dtype=int)  # array of turns
    fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(15, 7))

    # Plot from tracked & kicked particles
    axs["epsx"].plot(turns, kicktbt.epsilon_x, lw=2, label="Kicks")
    axs["epsy"].plot(turns, kicktbt.epsilon_y, lw=2, label="Kicks")
    axs["sigd"].plot(turns, kicktbt.sigma_delta, lw=2, label="Kicks")
    axs["bl"].plot(turns, kicktbt.bunch_length, lw=2, label="Kicks")

    # Plot from analytical values
    axs["epsx"].plot(turns, analyticaltbt.epsilon_x, lw=2.5, label="Analytical")
    axs["epsy"].plot(turns, analyticaltbt.epsilon_y, lw=2.5, label="Analytical")
    axs["sigd"].plot(turns, analyticaltbt.sigma_delta, lw=2.5, label="Analytical")
    axs["bl"].plot(turns, analyticaltbt.bunch_length, lw=2.5, label="Analytical")

    # Axes parameters
    axs["epsx"].set_ylabel(r"$\varepsilon_x$ [m]")
    axs["epsy"].set_ylabel(r"$\varepsilon_y$ [m]")
    axs["sigd"].set_ylabel(r"$\sigma_{\delta}$ [-]")
    axs["bl"].set_ylabel(r"Bunch length [m]")

    for axis in (axs["epsy"], axs["bl"]):
        axis.yaxis.set_label_position("right")
        axis.yaxis.tick_right()

    for axis in (axs["sigd"], axs["bl"]):
        axis.set_xlabel("Turn Number")

    for axis in axs.values():
        axis.yaxis.set_major_locator(plt.MaxNLocator(3))
        axis.legend(loc=9, ncols=4)

    fig.align_ylabels((axs["epsx"], axs["sigd"]))
    fig.align_ylabels((axs["epsy"], axs["bl"]))

    plt.tight_layout()
    if os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("CI") == "true":
        plt.savefig(f".plots/{filename}.pdf", dpi=300)
    # plt.savefig(f"{filename}.pdf", dpi=300)
