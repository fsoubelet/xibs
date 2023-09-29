import sys

from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt

from cpymad.madx import Madx
from lib.general_functions import BunchLength
from lib.IBSfunctions import NagaitsevIBS
from loguru import logger


@dataclass
class Records:
    """Dataclass to store (and update) important values through tracking."""

    epsilon_x: np.ndarray
    epsilon_y: np.ndarray
    sig_delta: np.ndarray
    bunch_length: np.ndarray


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--nturns", default=1000, type=int, show_default=True, help="Number of turns to track for.")
@click.option(
    "--sequence",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a file with the MAD-X sequence to load.",
)
@click.option(
    "--line",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a JSON file with the line to load.",
)
@click.option(
    "--outputdir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=".",
    show_default=True,
    help="Folder in which to write the output data.",
)
@click.option(
    "-v", "--verbose", count=True, default=0, help="The verbosity level, which determines the logging level."
)
def main(nturns: int, sequence: Path, line: Path, outputdir: Path, verbose: int) -> None:
    """Main program flow."""
    # ----- Adjust logging level ----- #
    set_logging_level(verbose)
    # ----- Create xtrack Line ----- #
    logger.info("Creating Line for tracking")
    if not sequence and not line:
        raise ValueError("At least one of the MAD-X sequence or line JSON is required")
    if sequence and not line:
        line: xt.Line = line_from_madx(sequence.absolute())
    else:  # we have the line
        if sequence:  # but also the sequence
            logger.debug("Both MAD-X sequence and JSON line file provided, prioritizing the line")
        line: xt.Line = line_from_json(line.absolute())

    # ----- Line optimizations ----- #
    logger.info("Optimizing line before tracking")
    line.remove_inactive_multipoles(inplace=True)
    line.remove_zero_length_drifts(inplace=True)
    line.merge_consecutive_drifts(inplace=True)
    line.merge_consecutive_multipoles(inplace=True)

    # ----- Choose context and build tracker ----- #
    logger.info("Compiling kernels")
    context = xo.ContextCpu()  # For CPU
    tracker = line.build_tracker(_context=context, extra_headers=["#define XTRACK_MULTIPOLE_NO_SYNRAD"])

    # ----- Prepare acceleration ----- #
    logger.info("Activating cavities")
    cavities = [element for element in line.elements if isinstance(element, xt.Cavity)]
    logger.debug(f"Found {len(cavities)} cavity to activate (lag = 180)")
    for cavity in cavities:
        cavity.lag = 180

    # ----- Particles and Twiss ----- #
    logger.info("Getting line Twiss and generating particles for tracking")
    bunch_intensity = 4.4e9
    sigma_z = 1.58e-3
    n_part = int(5e3)
    nemitt_x = 5.6644e-07
    nemitt_y = 3.7033e-09
    harmonic_number = 2852
    energy_loss = 0
    RF_voltage = 4.5
    IBS_step = 50.0  # recompute IBS kicks every IBS_Step turns
    p0 = xp.Particles(mass0=xp.ELECTRON_MASS_EV, q0=1, p0c=2.86e9)
    particles0 = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        particle_ref=p0,
        line=line,
    )
    twiss = line.twiss(particle_ref=p0)

    # ----- Main loop ----- #
    for model in ["kinetic", "simple", "analytical"]:
        logger.info(f"Performing tracking with IBS model '{model}'")

        # ----- Initialize particles and IBS ----- #
        logger.info("Initializing particles and IBS class")
        particles = particles0.copy()
        IBS = NagaitsevIBS()
        IBS.set_beam_parameters(particles)
        IBS.set_optic_functions(twiss)

        dt = 1.0 / IBS.frev  # consecutive turns/frev, used only for analytical,
        # emit = Evaluate_Sigma_Emit(10,20,10,20,20,20)
        # emit.define_bins_width(particles, tw)

        # ----- Initialze Records at first turn ----- #
        logger.trace("Initiating records")
        turn_by_turn = Records(
            epsilon_x=np.zeros(nturns, dtype=float),
            epsilon_y=np.zeros(nturns, dtype=float),
            sig_delta=np.zeros(nturns, dtype=float),
            bunch_length=np.zeros(nturns, dtype=float),
        )

        sig_x = np.std(particles.x[particles.state > 0])
        sig_y = np.std(particles.y[particles.state > 0])
        sig_delta = np.std(particles.delta[particles.state > 0])
        turn_by_turn.bunch_length[0] = np.std(particles.zeta[particles.state > 0])
        turn_by_turn.sig_delta[0] = sig_delta
        turn_by_turn.epsilon_x[0] = (sig_x**2 - (twiss["dx"][0] * sig_delta) ** 2) / twiss["betx"][0]
        turn_by_turn.epsilon_y[0] = sig_y**2 / twiss["bety"][0]

        # ----- Tracking ----- #
        logger.info(f"Starting tracking for {nturns} turns")
        for turn in range(1, nturns):  # start at 1 here as we initialized first entry from created particles
            logger.trace(f"Turn {turn} / {nturns}, computing particle properties")

            # ----- Calculate properties, different if kicks or analytical ----- #
            if model.lower() != "analytical":  # simple or kinetic
                sig_x = np.std(particles.x[particles.state > 0])
                sig_y = np.std(particles.y[particles.state > 0])
                sig_delta = np.std(particles.delta[particles.state > 0])
                turn_by_turn.bunch_length[turn] = np.std(particles.zeta[particles.state > 0])
                turn_by_turn.sig_delta[turn] = sig_delta
                turn_by_turn.epsilon_x[turn] = (sig_x**2 - (twiss["dx"][0] * sig_delta) ** 2) / twiss[
                    "betx"
                ][0]
                turn_by_turn.epsilon_y[turn] = sig_y**2 / twiss["bety"][0]

            else:  # here model is 'analytical', we rely on Nagaitsev to calculate evolutions
                if (turn % IBS_step == 0) or (turn == 1):  # recalculate the Nagaitsev integrals
                    logger.trace("Calculating Nagaitsev integrals")
                    IBS.calculate_integrals(  # this updates the IBS instance internal attributes
                        Emit_x=turn_by_turn.epsilon_x[turn - 1],
                        Emit_y=turn_by_turn.epsilon_y[turn - 1],
                        Sig_M=turn_by_turn.sig_delta[turn - 1],
                        BunchL=turn_by_turn.bunch_length[turn - 1],
                    )
                logger.trace("Determining emittances evolution from Nagaitsev integrals")
                emit_x, emit_y, sig_m = IBS.emit_evol(
                    Emit_x=turn_by_turn.epsilon_x[turn - 1],
                    Emit_y=turn_by_turn.epsilon_y[turn - 1],
                    Sig_M=turn_by_turn.sig_delta[turn - 1],
                    BunchL=turn_by_turn.bunch_length[turn - 1],
                    dt=dt,
                )
                logger.trace("Determining sigma_e and bunch_length from computed properties")
                sigma_e = sig_m * IBS.betar**2
                bunch_l = BunchLength(
                    IBS.Circu,
                    harmonic_number,
                    IBS.EnTot,
                    IBS.slip,
                    sigma_e,
                    IBS.betar,
                    RF_voltage * 1e-3,
                    energy_loss,
                    IBS.Ncharg,
                )
                logger.trace("Updating records")
                turn_by_turn.bunch_length[turn] = bunch_l
                turn_by_turn.sig_delta[turn] = sig_m
                turn_by_turn.epsilon_x[turn] = emit_x
                turn_by_turn.epsilon_y[turn] = emit_y

            # ----- Calculating kicks ----- #
            if (turn % IBS_step == 0) or (turn == 1):
                logger.debug(
                    f"Turn {turn} / {nturns}, {len(particles.x[particles.state > 0])} surviving particles"
                )
                logger.trace(f"Turn {turn} / {nturns}, re-calculating IBS {model.capitalize()} kick")
                if model.lower() == "simple":
                    IBS.calculate_simple_kick(particles)
                elif model.lower() == "kinetic":
                    IBS.calculate_kinetic_coefficients(particles)

            # ----- Applying relevant kick to particles ----- #
            if model.lower() == "simple":
                logger.trace("Applying simple kick")
                IBS.apply_simple_kick(particles)
            elif model.lower() == "kinetic":
                logger.trace("Applying kinetic kick")
                IBS.apply_kinetic_kick(particles)

            # ----- Track 1 turn through line ----- #
            line.track(particles)

        # ----- Saving data to file ----- #
        logger.debug("Saving recorded data to file")
        np.savez(
            outputdir / f"new_script_new_xsuite_{model.lower()}.npz",
            epsilon_x=turn_by_turn.epsilon_x,
            epsilon_y=turn_by_turn.epsilon_y,
            sig_delta=turn_by_turn.sig_delta,
            bunch_length=turn_by_turn.bunch_length,
        )


def line_from_madx(sequence_file: Path) -> xt.Line:
    """Function to load the line from MAD-X (hardcoded behavior)."""
    logger.debug("Loading line from MAD-X")
    n_slice_per_element = 4
    madx = Madx(stdout=False)
    madx.call(sequence_file)
    madx.command.beam(particle="positron", energy=2.86, bunched=True)
    madx.command.use(sequence="RING")
    madx.command.select(flag="MAKETHIN", slice_=n_slice_per_element, thick=False)
    madx.command.select(flag="MAKETHIN", pattern="wig", slice_=1)
    madx.command.makethin(sequence="RING", makedipedge=True)
    madx.command.use(sequence="RING")
    logger.debug("Converting MAD-X sequence to xtrack line")
    return xt.Line.from_madx_sequence(madx.sequence.ring)


def line_from_json(json_file: Path) -> xt.Line:
    """Function to load the line directly from a JSON file."""
    logger.debug("Loading line from MAD-X")
    return xt.Line.from_json(json_file)


def set_logging_level(verbosity: int) -> None:
    logger.remove()
    verbosity = 3 if verbosity > 3 else verbosity  # cap to 3, as we can't be more verbose
    verbosity_to_level = {0: "WARNING", 1: "INFO", 2: "DEBUG", 3: "TRACE"}
    logger.add(sys.stdout, level=verbosity_to_level[verbosity])


if __name__ == "__main__":
    main()
