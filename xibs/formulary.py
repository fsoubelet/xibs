"""
.. _xibs-utils:

Formulary
---------

Module with commonly used formulae to compute quantities of interest needed in the rest of the package.
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

import numpy as np

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


# This is BunchLength from Michail's code, in general_functions.py
# The arguments used to be Circumferance, Harmonic_Num, Energy_total, SlipF, Sigma_E, beta_rel, RF_Voltage, Energy_loss, Z
def bunch_length(
    circumference: float,
    harmonic_number: int,
    total_energy_GeV: float,
    slip_factor: float,
    sigma_e: float,
    beta_rel: float,
    rf_voltage: float,
    energy_loss: float,
    particle_charge: int,
) -> float:
    """Analytical calculation for bunch length for protons / electrons.
    
    This is a linear approximation which assumes that particles are in the center of the bucket only / mostly.
    ~~~ from Wiedermanns book ~~~

    TODO: figure out exactly the formula that was implemented here and reference / document it in the docstring.
    Args:
        circumference (float): machine circumference in [m].
        harmonic_number (int): harmonic number of the RF system.
        total_energy_GeV (float): total energy of the simulated particles in [GeV].
        slip_factor (float): slip factor of the machine.
        sigma_e (float): energy spread of the particles? TODO: check with Michail and in Wiedermann book.
        beta_rel (float): relativistic beta of the simulated particles.
        rf_voltage (float): RF voltage of the machine's cavities in [???]. TODO: check with Michail for the units.
        energy_loss (float): ??? in [???]. TODO: check with Michail and in Wiedermann book.
        particle_charge (int): elementary particle charge, in # of Coulomb charges (for
            instance 1 for electron or proton).
    
    Returns:
        The analytically calculated bunch length in [m].
    """
    return (
        sigma_e
        * circumference
        * np.sqrt(
            abs(slip_factor)
            * total_energy_GeV
            / (
                2
                * np.pi
                * beta_rel
                * harmonic_number
                * np.sqrt(particle_charge**2 * rf_voltage**2 - energy_loss**2)
            )
        )
    )
