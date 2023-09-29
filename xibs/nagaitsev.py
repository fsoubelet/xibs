"""
.. _xibs-nagaitsev:

Nagaitsev Integrals
-------------------

Module with functionality to perform analytical calculations according to Nagaitsev's formalism.
A user-facing class is provided which computes the Nagaitsev integrals based on beam optics.
The formalism from which formulas and calculations are implemented can be found in :cite:p:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

from dataclasses import dataclass, field

import numpy as np

from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.interpolate import interp1d


@dataclass
class NagaitsevIntegrals:
    """Container dataclass for Nagaitsev integrals results."""

    Ix: float
    Iy: float
    Iz: float


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


# TODO: maybe this should self-init from an xtrack.twiss.TwissTable object?
@dataclass
class OpticsParameters:
    """Container dataclass for necessary optics parameters.

    Args:
        s (ArrayLike): longitudinal positions of the machine elements in [m]. If
            initiating atfer a TWISS result, this is `twiss.s`.
        circumference (float): machine circumference in [m]. If initiating after
            a TWISS result, this is `twiss.s[-1]`.
        slip_factor (float): slip factor of the machine. If initiating after a line
            twiss from ``xsuite``, this is `twiss["slip_factor"]`.
        revolution_frequency (float): revolution frequency of the machine in [Hz].
        betx (ArrayLike): horizontal beta functions in [m]. If initiating after a
            TWISS result, this is `twiss.betx`.
        bety (ArrayLike): vertical beta functions in [m]. If initiating after a
            TWISS result, this is `twiss.bety`.
        alfx (ArrayLike): horizontal alpha functions. If initiating after a TWISS
            result, this is `twiss.alfx`.
        alfy (ArrayLike): vertical alpha functions. If initiating after a TWISS
            result, this is `twiss.alfy`.
        dx (ArrayLike): horizontal dispersion functions in [m]. If initiating after
            a TWISS result, this is `twiss.dx`.
        dy (ArrayLike): vertical dispersion functions in [m]. If initiating after
            a TWISS result, this is `twiss.dy`.
        dpx (ArrayLike): horizontal dispersion derivative functions. If initiating
            after a TWISS result, this is `twiss.dpx`.
        dpy (ArrayLike): vertical dispersion derivative functions. If initiating
            after a TWISS result, this is `twiss.dpy`.
    """

    s: ArrayLike
    circumference: float
    slip_factor: float
    revolution_frequency: float
    betx: ArrayLike
    bety: ArrayLike
    alfx: ArrayLike
    alfy: ArrayLike
    dx: ArrayLike  # this is eta_x in michail's code
    dy: ArrayLike  # this is eta_y in michail's code
    dpx: ArrayLike  # this is eta_dx in michail's code
    dpy: ArrayLike  # this is eta_dy in michail's code
    bx_bar: ArrayLike = field(init=False)
    by_bar: ArrayLike = field(init=False)
    dx_bar: ArrayLike = field(init=False)
    dy_bar: ArrayLike = field(init=False)

    def __post_init__(self):
        # Interpolated functions for the calculation below
        bxb = interp1d(self.s, self.betx)
        byb = interp1d(self.s, self.bety)
        dxb = interp1d(self.s, self.dx)
        dyb = interp1d(self.s, self.dy)
        # Below is the average beta and dispersion functions - better here than a simple
        # np.mean calculation because the latter doesn't take in consideration element lengths
        # These are ONLY USED in the CoulogConst function -> can be extracted there, in time
        self.bx_bar = quad(bxb, self.s[0], self.s[-1])[0] / self.circumference
        self.by_bar = quad(byb, self.s[0], self.s[-1])[0] / self.circumference
        self.dx_bar = quad(dxb, self.s[0], self.s[-1])[0] / self.circumference
        self.dy_bar = quad(dyb, self.s[0], self.s[-1])[0] / self.circumference


class Nagaitsev:
    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        self.beam_parameters = beam_params
        self.optics = optics
