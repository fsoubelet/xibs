"""
.. _xibs-utils:

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
