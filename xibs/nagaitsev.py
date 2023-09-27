"""
.. _xibs-nagaitsev:

Nagaitsev Integrals
-------------------

Module with functionality to perform analytical calculations according to Nagaitsev's formalism.
A user-facing class is provided which computes the Nagaitsev integrals based on beam optics.
The formalism from which formulas and calculations are implemented can be found in :cite:p:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.
"""
from dataclasses import dataclass

import numpy as np

from numpy.typing import ArrayLike


@dataclass
class NagaitsevIntegrals:
    """Container dataclass for Nagaitsev integrals results."""

    Ix: float
    Iy: float
    Iz: float


@dataclass
class BeamParameters:
    """Container dataclass for necessary beam parameters."""

    n_part: int
    particle_charge: int  # this is particles.q0 from xsuite
    particle_mass_GeV: float  # in GeV, so this is particles.mass0 * 1e-9 from xsuite
    particle_classical_radius: float  # clasiscal radius, can be taken from scipy physical constants or computed manually by user
    total_energy_GeV: float  # why does Michail calculate as np.sqrt(particles.p0c[0] ** 2 + particles.mass0**2) * 1e-9 ??
    gamme_rel: float  # relativistic gamma, this is particles.gamma0[0] from xsuite
    beta_rel: float  # relativistic beta, this is particles.beta0[0] from xsuite
    c_rad: float  # classical radius, to be computed from the previous


@dataclass
class OpticsParameters:
    """Container dataclass for necessary optics parameters."""

    circumference: float
    slip_factor: float
    frev: float
    s: ArrayLike
    betx: ArrayLike
    bety: ArrayLike
    alfx: ArrayLike
    alfy: ArrayLike
    etax: ArrayLike
    etay: ArrayLike
    etadx: ArrayLike
    etady: ArrayLike


class Nagaitsev:
    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        self.beam_parameters = beam_params
        self.optics = optics
