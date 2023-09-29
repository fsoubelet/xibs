"""
.. _xibs-nagaitsev:

Nagaitsev Integrals
-------------------

Module with functionality to perform analytical calculations according to Nagaitsev's formalism.
A user-facing class is provided which computes the Nagaitsev integrals based on beam optics.
The formalism from which formulas and calculations are implemented can be found in :cite:p:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

from dataclasses import InitVar, dataclass, field

import numpy as np

from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.interpolate import interp1d

# ----- Dataclasses for inputs to Nagaitsev class ----- #


# TODO: maybe this should self-init from an xpart.Particles object?
@dataclass
class BeamParameters:
    """Container dataclass for necessary beam parameters.

    Args:
        n_part (int): number of simulated particles.
        particle_charge (int): particle charge in elementary. If generating particles
            with ``xsuite``, this is `particles.q0`.
        particle_mass_GeV (float): particle mass in [GeV]. If generating particles with
            ``xsuite``, this is `particles.mass0 * 1e-9`.
        particle_classical_radius (float): particle classical radius in [???]. It can be
            taken from constants in ``scipy`` or computed manually by the user.
        total_energy_GeV (float): total energy of the simulated particles in [GeV]. If
            generating particles with ``xsuite``, this is
            `np.sqrt(particles.p0c[0] ** 2 + particles.mass0**2) * 1e-9`.
        gamma_rel (float): relativistic gamma of the simulated particles. If generating
            particles with ``xsuite``, this is `particles.gamma0[0]`.
        beta_rel (float): relativistic beta of the simulated particles. If generating
            particles with ``xsuite``, this is `particles.beta0[0]`.
        c_rad (float): classical radius of the simulated particles. If generating
            particles with ``xsuite``, this is obtained with `particles.get_classical_particle_radius0()`.
    """

    n_part: int
    particle_charge: int  # this is particles.q0 from xsuite
    particle_mass_GeV: float  # in GeV, so this is particles.mass0 * 1e-9 from xsuite
    particle_classical_radius: float  # clasiscal radius, can be taken from scipy physical constants or computed manually by user
    total_energy_GeV: float  # why does Michail calculate as np.sqrt(particles.p0c[0] ** 2 + particles.mass0**2) * 1e-9 ??
    gamma_rel: float  # relativistic gamma, this is particles.gamma0[0] from xsuite
    beta_rel: float  # relativistic beta, this is particles.beta0[0] from xsuite
    c_rad: float  # classical radius, to be computed from the previous


@dataclass
class OpticsParameters:
    """Container dataclass for necessary optics parameters. It is initiated from
    the results of a ``TWISS`` command with an ``xsuite``.

    Args:
        twiss (xtrack.twiss.TwissTable): the resulting table of a ``TWISS`` call
            on the line in ``xsuite``.
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
    twiss: InitVar["xtrack.twiss.TwissTable"]  # Needed to initialize, almost all else is derived from there
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
    # Below are ONLY USED in the CoulogConst function -> extract? Kept private for now
    _bx_bar: ArrayLike = field(init=False)
    _by_bar: ArrayLike = field(init=False)
    _dx_bar: ArrayLike = field(init=False)
    _dy_bar: ArrayLike = field(init=False)

    def __post_init__(self, twiss: "xtrack.twiss.TwissTable"):
        # Attributes derived from the TwissTable
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
        # Interpolated beta and dispersion functions for the calculation below
        _bxb = interp1d(self.s, self.betx)
        _byb = interp1d(self.s, self.bety)
        _dxb = interp1d(self.s, self.dx)
        _dyb = interp1d(self.s, self.dy)
        # Computing "average" of these functions - better here than a simple np.mean
        # calculation because the latter doesn't take in consideration element lengths
        self._bx_bar = quad(_bxb, self.s[0], self.s[-1])[0] / self.circumference
        self._by_bar = quad(_byb, self.s[0], self.s[-1])[0] / self.circumference
        self._dx_bar = quad(_dxb, self.s[0], self.s[-1])[0] / self.circumference
        self._dy_bar = quad(_dyb, self.s[0], self.s[-1])[0] / self.circumference


# ----- Dataclasses to store results ----- #


@dataclass
class NagaitsevIntegrals:
    """Container dataclass for Nagaitsev integrals results.

    Args:
        Ix (float): horizontal Nagaitsev integral.
        Iy (float): vertical Nagaitsev integral.
        Iz (float): longitudinal Nagaitsev integral.
    """

    Ix: float
    Iy: float
    Iz: float


@dataclass
class IBSGrowthRates:
    """Container dataclass for IBS growth rates results.

    Args:
        Tx (float): horizontal IBS growth rate.
        Ty (float): vertical IBS growth rate.
        Tz (float): longitudinal IBS growth rate.
    """

    Tx: float
    Ty: float
    Tz: float


# ----- Main class to compute Nagaitsev integrals and IBS growth rates ----- #


class Nagaitsev:
    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        self.beam_parameters = beam_params
        self.optics = optics
