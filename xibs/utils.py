"""
.. _xibs-utils:

Utilities
---------

Module with convenience functions to compute quantities of interest needed in the rest of the package.
"""
from numpy.typing import ArrayLike


def phi(beta: ArrayLike, alpha: ArrayLike, dx: ArrayLike, dpx: ArrayLike) -> ArrayLike:
    """Phi parameter of Eq (15) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.

    TODO: figure out exactly who's what here, and why the calculation is not as Eq (15)
    Args:
        beta (ArrayLike): beta-functions through the machine.
        alpha (ArrayLike): alpha-functions through the machine.
        dx (ArrayLike): dispersion function through the machine.
        dpx (ArrayLike): dpx function through the machine.

    Returns:
        An array of phi values through the machine.
    """
    return dpx + alpha * dx / beta
