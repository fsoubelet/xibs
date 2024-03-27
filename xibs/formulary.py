"""
.. _xibs-formulary:

Formulary
---------

Module with commonly used formulae to compute quantities of interest needed in the rest of the package.
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

import logging

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


# ----- Some helpers on xtrack.Particles objects ----- #


def _bunch_length(particles: "xtrack.Particles") -> float:  # noqa: F821
    """Get the bunch length from the particles."""
    nplike = particles._context.nplike_lib
    return nplike.std(particles.zeta[particles.state > 0])


def _sigma_delta(particles: "xtrack.Particles") -> float:  # noqa: F821
    """Get the standard deviation of the momentum spread from the particles."""
    nplike = particles._context.nplike_lib
    return nplike.std(particles.delta[particles.state > 0])


def _sigma_x(particles: "xtrack.Particles") -> float:  # noqa: F821
    """Get the horizontal coordinate standard deviation from the particles."""
    nplike = particles._context.nplike_lib
    return nplike.std(particles.x[particles.state > 0])


def _geom_epsx(particles: "xtrack.Particles", betx: float, dx: float) -> float:  # noqa: F821
    """
    Horizontal geometric emittance at a location in the machine, for the beta
    and dispersion functions at this location.
    """
    sigma_x = _sigma_x(particles)
    sig_delta = _sigma_delta(particles)
    return (sigma_x**2 - (dx * sig_delta) ** 2) / betx


def _sigma_y(particles: "xtrack.Particles") -> float:  # noqa: F821
    """Get the vertical coordinate standard deviation from the particles."""
    nplike = particles._context.nplike_lib
    return nplike.std(particles.y[particles.state > 0])


def _geom_epsy(particles: "xtrack.Particles", bety: float, dy: float) -> float:  # noqa: F821
    """
    Vertical geometric emittance at a location in the machine, for the beta
    and dispersion functions at this location.
    """
    sigma_y = _sigma_y(particles)
    sig_delta = _sigma_delta(particles)
    return (sigma_y**2 - (dy * sig_delta) ** 2) / bety


# ----- Some helpers on simple calculations ----- #


def _percent_change(initial_value: float, final_value: float) -> float:
    """Calculate the percentage change between two values."""
    return 100 * (final_value - initial_value) / initial_value
