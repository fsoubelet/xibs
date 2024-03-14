"""
.. _xibs-formulary:

Formulary
---------

Module with commonly used formulae to compute quantities of interest needed in the rest of the package.
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

import logging

import numpy as np

from numpy.typing import ArrayLike

LOGGER = logging.getLogger(__name__)


def phi(beta: ArrayLike, alpha: ArrayLike, dx: ArrayLike, dpx: ArrayLike) -> ArrayLike:
    """
    .. versionadded:: 0.2.0

    Phi parameter of Eq (15) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.

    Args:
        beta (ArrayLike): beta-functions through the machine.
        alpha (ArrayLike): alpha-functions through the machine.
        dx (ArrayLike): dispersion function through the machine.
        dpx (ArrayLike): dpx function through the machine.

    Returns:
        An array of phi values through the machine.
    """
    return dpx + alpha * dx / beta


# ----- Some helpers ----- #


def _bunch_length(particles: "xtrack.Particles") -> float:
    """Get the bunch length from the particles."""
    nplike = particles._context.nplike_lib
    return nplike.std(particles.zeta[particles.state > 0])


def _sigma_delta(particles: "xtrack.Particles") -> float:
    """Get the standard deviation of the momentum spread from the particles."""
    nplike = particles._context.nplike_lib
    return nplike.std(particles.delta[particles.state > 0])


def _geom_epsx(particles: "xtrack.Particles", betx: float, dx: float) -> float:
    """
    Horizontal geometric emittance at a location in the machine, for the beta
    and dispersion functions at this location.
    """
    nplike = particles._context.nplike_lib
    sigma_x = nplike.std(particles.x[particles.state > 0])
    sig_delta = _sigma_delta(particles)
    return (sigma_x**2 - (dx * sig_delta) ** 2) / betx


def _geom_epsy(particles: "xtrack.Particles", bety: float, dy: float) -> float:
    """
    Vertical geometric emittance at a location in the machine, for the beta
    and dispersion functions at this location.
    """
    nplike = particles._context.nplike_lib
    sigma_y = nplike.std(particles.y[particles.state > 0])
    sig_delta = _sigma_delta(particles)
    return (sigma_y**2 - (dy * sig_delta) ** 2) / bety
