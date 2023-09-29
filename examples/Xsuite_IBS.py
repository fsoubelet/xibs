import sys

from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt

from cpymad.madx import Madx
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
    "--model",
    type=click.Choice(["Simple", "Kinetic", "Analytical"], case_sensitive=False),
    show_default=True,
    default="Simple",
    help="The IBS model to use. Simple of kinetic kicks, or analytical model based on Nagaitsev's integrals.",
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
def main(nturns: int, sequence: Path, line: Path, model: str, outputdir: Path, verbose: int) -> None:
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

    # ----- Choose context and build tracker ----- #
    logger.info("Compiling kernels")
    context = xo.ContextCpu()  # For CPU
    tracker = line.build_tracker(_context=context)

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

    # ----- Initialize IBS ----- #
    logger.info("Initializing IBS class")
    IBS = NagaitsevIBS()
    IBS.set_beam_parameters(particles)
    IBS.set_optic_functions(twiss)

    # ----- Initialze Records ----- #
    turn_by_turn = Records(
        epsilon_x=np.zeros(nturns, dtype=float),
        epsilon_y=np.zeros(nturns, dtype=float),
        sig_delta=np.zeros(nturns, dtype=float),
        bunch_length=np.zeros(nturns, dtype=float),
    )

    # ----- Tracking ----- #
    logger.info(f"Stating tracking for {nturns} turns")
    for turn in range(nturns):
        logger.trace(f"Turn {turn} / {nturns}, computing particle properties")
        sig_x = np.std(particles.x[particles.state > 0])
        sig_y = np.std(particles.y[particles.state > 0])
        sig_delta = np.std(particles.delta[particles.state > 0])
        turn_by_turn.bunch_length[turn] = np.std(particles.zeta[particles.state > 0])
        turn_by_turn.sig_delta[turn] = sig_delta
        turn_by_turn.epsilon_x[turn] = (sig_x**2 - (twiss["dx"][0] * sig_delta) ** 2) / twiss["betx"][0]
        turn_by_turn.epsilon_y[turn] = sig_y**2 / twiss["bety"][0]
        # turn_by_turn.epsilon_x[turn], turn_by_turn.epsilon_y[turn], turn_by_turn.sig_delta[turn], turn_by_turn.bunch_length[turn] = emit.eval_emits(particles)

        if turn % 50 == 0:
            logger.debug(
                f"Turn {turn} / {nturns}, {len(particles.x[particles.state > 0])} surviving particles"
            )
            logger.trace(f"At turn {turn}, re-calculating IBS {model.capitalize()} kick")
            if model.lower() == "simple":
                IBS.calculate_simple_kick(particles)
            elif model.lower() == "kinetic":
                IBS.calculate_kinetic_coefficients(particles)

        # ----- Applying relevant kick to particles ----- #
        if model.lower() == "simple":
            IBS.apply_simple_kick(particles)
        elif model.lower() == "kinetic":
            IBS.apply_kinetic_kick(particles)

        # ----- Track 1 turn through line ----- #
        line.track(particles)

    # ----- Saving data to file ----- #
    logger.debug("Saving recorded data to file")
    np.savez(
        outputdir / f"xsuite_{model.lower()}.npz",
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
