"""
.. _xibs-inputs:

Input Data Structures
---------------------

Module with container dataclasses to encompass expected necessary inputs .
Various parts of ``xibs`` perform calculations based on beam and optics parameters.

The former is encompassed in a `BeamParameters` dataclass which is initiated from a generated `xpart.Particles` object.
The latter is encompassed in an `OpticsParameters` dataclass which is initiated from the result of a ``TWISS`` call on the line.
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

from dataclasses import InitVar, dataclass, field
from logging import getLogger

import numpy as np

from numpy.typing import ArrayLike

LOGGER = getLogger(__name__)

# ----- Dataclasses for inputs to Nagaitsev class ----- #


@dataclass
class BeamParameters:
    """Container dataclass for necessary beam parameters. It is initiated from
    the `xpart.Particles` object to track in your line with ``xsuite``.

    Args:
        particles (xpart.Particles): the generated particles to be tracked or used
            in the line. This is an init-only parameter used for instanciation and
            it will not be kept in the instance's attributes.

    Attributes:
        n_part (int): number of simulated particles.
        particle_charge (int): elementary particle charge, in # of Coulomb charges (for
            instance 1 for electron or proton).
        particle_mass_GeV (float): particle mass in [GeV].
        total_energy_GeV (float): total energy of the simulated particles in [GeV].
        gamma_rel (float): relativistic gamma of the simulated particles.
        beta_rel (float): relativistic beta of the simulated particles.
        particle_classical_radius_m (float): the particles' classical radius in [m].
    """

    # ----- To be provided at initialization ----- #
    particles: InitVar["xpart.Particles"]  # Almost all is derived from there, this is not kept!
    # ----- Below are attributes derived from the Particles object ----- #
    # The following are Npart, Ncharg, E_rest, EnTot, gammar, betar and c_rad in Michail's code
    n_part: int = field(init=False)
    particle_charge: int = field(init=False)
    particle_mass_GeV: float = field(init=False)
    total_energy_GeV: float = field(init=False)
    gamma_rel: float = field(init=False)
    beta_rel: float = field(init=False)
    particle_classical_radius_m: float = field(init=False)

    def __post_init__(self, particles: "xpart.Particles"):
        # Attributes derived from the Particles object
        LOGGER.debug("Initializing BeamParameters from Particles object")
        self.n_part = particles.weight[0] * particles.gamma0.shape[0]
        self.particle_charge = particles.q0
        self.particle_mass_GeV = particles.mass0 * 1e-9
        self.total_energy_GeV = np.sqrt(particles.p0c[0] ** 2 + particles.mass0**2) * 1e-9
        self.gamma_rel = particles.gamma0[0]
        self.beta_rel = particles.beta0[0]
        self.particle_classical_radius_m = particles.get_classical_particle_radius0()


@dataclass
class OpticsParameters:
    """Container dataclass for necessary optics parameters. It is initiated from
    the results of a ``TWISS`` command with an ``xsuite``.

    Args:
        twiss (xtrack.twiss.TwissTable): the resulting table of a ``TWISS`` call
            on the line in ``xsuite``. This is an init-only parameter used for
            instanciation and it will not be kept in the instance's attributes.
        revolution_frequency (float): revolution frequency of the machine in [Hz].
            If initiating after from ``xsuite`` elements, this can be obtained with
            `particle_ref.beta0[0] * scipy.constants.c / twiss.s[-1]`.

    Attributes:
        revolution_frequency (float): revolution frequency of the machine in [Hz].
        s (ArrayLike): longitudinal positions of the machine elements in [m].
        circumference (float): machine circumference in [m].
        slip_factor (float): slip factor of the machine.
        betx (ArrayLike): horizontal beta functions in [m].
        bety (ArrayLike): vertical beta functions in [m].
        alfx (ArrayLike): horizontal alpha functions.
        alfy (ArrayLike): vertical alpha functions.
        dx (ArrayLike): horizontal dispersion functions in [m].
        dy (ArrayLike): vertical dispersion functions in [m].
        dpx (ArrayLike): horizontal dispersion (d px / d delta).
        dpy (ArrayLike): horizontal dispersion (d px / d delta).
    """

    # ----- To be provided at initialization ----- #
    twiss: InitVar["xtrack.twiss.TwissTable"]  # Almost all is derived from there, this is not kept!
    revolution_frequency: float
    # ----- Below are attributes derived from the twiss table ----- #
    s: ArrayLike = field(init=False)
    circumference: float = field(init=False)
    slip_factor: float = field(init=False)
    betx: ArrayLike = field(init=False)
    bety: ArrayLike = field(init=False)
    alfx: ArrayLike = field(init=False)
    alfy: ArrayLike = field(init=False)
    # The following four are eta_x, eta_y, eta_dx and eta_dy in michail's code
    dx: ArrayLike = field(init=False)
    dy: ArrayLike = field(init=False)
    dpx: ArrayLike = field(init=False)
    dpy: ArrayLike = field(init=False)

    def __post_init__(self, twiss: "xtrack.twiss.TwissTable"):
        # Attributes derived from the TwissTable
        LOGGER.debug("Initializing OpticsParameters from TwissTable object")
        self.s = twiss.s
        self.circumference = twiss.s[-1]
        self.slip_factor = twiss["slip_factor"]
        self.betx = twiss.betx
        self.bety = twiss.bety
        self.alfx = twiss.alfx
        self.alfy = twiss.alfy
        self.dx = twiss.dx
        self.dy = twiss.dy
        self.dpx = twiss.dpx
        self.dpy = twiss.dpy
