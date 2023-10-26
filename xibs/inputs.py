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
from typing import Optional, Union

import numpy as np

from numpy.typing import ArrayLike
from scipy.constants import c

LOGGER = getLogger(__name__)

# ----- Dataclasses for inputs to Nagaitsev class ----- #


@dataclass
class BeamParameters:
    """
    .. versionadded:: 0.2.0

    Container dataclass for necessary beam parameters. It is initiated from
    the `xpart.Particles` object to track in your line with ``xsuite``.

    Args:
        particles (xpart.Particles): the generated particles to be tracked or used
            in the line. This is an init-only parameter used for instanciation and
            it will not be kept in the instance's attributes.

    Attributes:
        n_part (int): number of simulated particles.
        particle_charge (int): elementary particle charge, in # of Coulomb charges
            (for instance 1 for electron or proton).
        particle_mass_GeV (float): particle mass in [GeV].
        total_energy_GeV (float): total energy of the simulated particles in [GeV].
        gamma_rel (float): relativistic gamma of the simulated particles.
        beta_rel (float): relativistic beta of the simulated particles.
        particle_classical_radius_m (float): the particles' classical radius in [m].
    """

    # ----- To be provided at initialization ----- #
    particles: InitVar["xpart.Particles"]  # Almost all is derived from there, this is not kept!
    # ----- Below are attributes derived from the Particles object ----- #
    # The following are Npart, Ncharg, E_rest, EnTot, gammar, betar and c_rad in Michalis's code
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
    """
    .. versionadded:: 0.2.0

    Container dataclass for necessary optics parameters. It is initiated from the results
    of a `.twiss()` command with an `xtrack.Line`, or result of a ``TWISS`` call in
    ``MAD-X`` as a dataframe (as given by ``cpymad`` by default).

    Args:
        twiss (Union["xtrack.twiss.TwissTable", pd.DataFrame]): the resulting `TwissTable`
            of a `.twiss()` call on the `xtrack.Line`, **or** the ``TWISS`` table of the
            ``MAD-X`` sequence as a `pandas.DataFrame`. Lowercase keys are expected. If
            initiating from a ``MAD-X`` twiss, the next two arguments are required. This
            is an init-only parameter used for instanciation and it will not be kept in
            the instance's attributes.
        _slipfactor (Optional[float]): the slip factor for the machine. Only required if
            the ``twiss`` argument is a ``MAD-X`` twiss dataframe.
        _frev_hz (Optional[float]): the revolution frequency for the machine in [Hz].
            Only required if the ``twiss`` argument is a ``MAD-X`` twiss dataframe.


    Attributes:
        s (ArrayLike): longitudinal positions of the machine elements in [m].
        circumference (float): machine circumference in [m].
        slip_factor (float): slip factor of the machine.
        revolution_frequency (float): revolution frequency of the machine in [Hz].
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
    twiss: InitVar[  # Almost all is derived from there, this is not kept!
        Union["xtrack.twiss.TwissTable", "pandas.DataFrame"]
    ]
    _slipfactor: InitVar[Optional[float]] = None
    _frev_hz: InitVar[Optional[float]] = None
    # ----- Below are attributes derived from the twiss table ----- #
    s: ArrayLike = field(init=False)
    circumference: float = field(init=False)
    slip_factor: float = field(init=False)
    revolution_frequency: float = field(init=False)
    betx: ArrayLike = field(init=False)
    bety: ArrayLike = field(init=False)
    alfx: ArrayLike = field(init=False)
    alfy: ArrayLike = field(init=False)
    # The following four are eta_x, eta_y, eta_dx and eta_dy in Michalis's code
    dx: ArrayLike = field(init=False)
    dy: ArrayLike = field(init=False)
    dpx: ArrayLike = field(init=False)
    dpy: ArrayLike = field(init=False)

    def __post_init__(
        self,
        twiss: Union["xtrack.twiss.TwissTable", "pandas.DataFrame"],
        # The following are only needed if we instanciate from MAD-X twiss
        _slipfactor: Optional[float] = None,
        _frev_hz: Optional[float] = None,
    ):
        # Attributes derived from the TwissTable
        self.s = np.array(twiss.s)
        self.circumference = twiss.s[-1]
        self.betx = np.array(twiss.betx)
        self.bety = np.array(twiss.bety)
        self.alfx = np.array(twiss.alfx)
        self.alfy = np.array(twiss.alfy)
        self.dx = np.array(twiss.dx)
        self.dy = np.array(twiss.dy)
        self.dpx = np.array(twiss.dpx)
        self.dpy = np.array(twiss.dpy)
        try:  # assume we have been given an xtrack.TwissTable
            self.slip_factor = twiss["slip_factor"]
            self.revolution_frequency = twiss.beta0 * c / self.circumference
            LOGGER.debug("Initialized OpticsParameters from TwissTable object")
        except KeyError:  # then it is a MAD-X twiss dataframe and we need these 2 provided
            self.slip_factor = _slipfactor
            self.revolution_frequency = _frev_hz
            LOGGER.debug("Initialized OpticsParameters from MAD-X Twiss dataframe")
