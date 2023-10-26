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


# This is BunchLength from Michalis's code, in general_functions.py
# The arguments used to be Circumferance, Harmonic_Num, Energy_total, SlipF, Sigma_E, beta_rel, RF_Voltage, Energy_loss, Z
def bunch_length(
    circumference: float,
    harmonic_number: int,
    total_energy_GeV: float,
    slip_factor: float,
    sigma_e: float,
    beta_rel: float,
    rf_voltage_GV: float,
    energy_loss_GeV: float,
    particle_charge: int,
) -> float:
    """
    .. versionadded:: 0.2.0

    Analytical calculation for bunch length for protons / electrons (from Wiedermann's book).
    This is a linear approximation which assumes that particles are in the center of the bucket
    only / mostly.

    .. todo::
        Figure out exactly the formula that was implemented here and reference / document it in the docstring.

    Args:
        circumference (float): machine circumference in [m].
        harmonic_number (int): harmonic number of the RF system.
        total_energy_GeV (float): total energy of the simulated particles in [GeV].
        slip_factor (float): slip factor of the machine.
        sigma_e (float): relative energy spread of the particles.
        beta_rel (float): relativistic beta of the simulated particles.
        rf_voltage_GV (float): RF voltage of the machine's cavities in [GV].
        energy_loss_GeV (float): The turn-by-turn oarticle energy loss in [GeV].
        particle_charge (int): elementary particle charge, in # of Coulomb charges
            (for instance 1 for electron or proton).

    Returns:
        The analytically calculated bunch length in [m].
    """
    # fmt: off
    return (
        sigma_e
        * circumference
        * np.sqrt(
            abs(slip_factor) * total_energy_GeV
            / (2 * np.pi * beta_rel * harmonic_number * np.sqrt(particle_charge**2 * rf_voltage_GV**2 - energy_loss_GeV**2))
        )
    )
    # fmt: on


# This is EnergySpread from Michalis's code, in general_functions.py
# The arguments used to be Circumferance, Harmonic_Num, Energy_total, SlipF, BL, beta_rel, RF_Voltage, Energy_loss, Z
def energy_spread(
    circumference: float,
    harmonic_number: int,
    total_energy_GeV: float,
    slip_factor: float,
    bunch_length: float,
    beta_rel: float,
    rf_voltage_GV: float,
    energy_loss_GeV: float,
    particle_charge: int,
) -> float:
    """
    .. versionadded:: 0.2.0

    Counterpart of the `bunch_length` function, analytically calculates for bunch length for
    protons / electrons (from Wiedermann's book). The same caveats than in `bunch_length` apply
    here.

    Args:
        circumference (float): machine circumference in [m].
        harmonic_number (int): harmonic number of the RF system.
        total_energy_GeV (float): total energy of the simulated particles in [GeV].
        slip_factor (float): slip factor of the machine.
        bunch_length (float): bunch length in [m].
        beta_rel (float): relativistic beta of the simulated particles.
        rf_voltage_GV (float): RF voltage of the machine's cavities in [GV].
        energy_loss_GeV (float): The turn-by-turn oarticle energy loss in [GeV].
        particle_charge (int): elementary particle charge, in # of Coulomb charges
            (for instance 1 for electron or proton).

    Returns:
        The analytically calculated dimensionless energy spread for the particle bunch.
    """
    # fmt: off
    return bunch_length / (
        circumference
        * np.sqrt(
            abs(slip_factor) * total_energy_GeV
            / (2 * np.pi * beta_rel * harmonic_number * np.sqrt(particle_charge**2 * rf_voltage_GV**2 - energy_loss_GeV**2))
        )
    )
    # fmt: on


# This is ion_BunchLength from Michalis's code, in general_functions.py
# The arguments used to be Circumferance, Harmonic_Num, Energy_total, SlipF, Sigma_E, beta_rel, RF_Voltage, Z
def ion_bunch_length(
    circumference: float,
    harmonic_number: int,
    total_energy_GeV: float,
    slip_factor: float,
    sigma_e: float,
    beta_rel: float,
    rf_voltage_GV: float,
    particle_charge: int,
):
    """
    .. versionadded:: 0.2.0

    Analytical calculation for bunch length for ions (from Wiedermann's book). This calculation
    does not work too well if the bucket is full. Implementation so far as I can tell is as used
    in the scripts for LEIR for ions (some MAD-X script that LEIR studies were using for IBS).

    .. todo::
        Figure out exactly the formula that was implemented here and reference / document it in the docstring.

    Args:
        circumference (float): machine circumference in [m].
        harmonic_number (int): harmonic number of the RF system.
        total_energy_GeV (float): total energy of the simulated particles in [GeV].
        slip_factor (float): slip factor of the machine.
        sigma_e (float): relative energy spread of the particles.
        beta_rel (float): relativistic beta of the simulated particles.
        rf_voltage_GV (float): RF voltage of the machine's cavities in [GV].
        particle_charge (int): elementary particle charge, in # of Coulomb charges (for
            instance 1 for electron or proton).

    Returns:
        The analytically calculated bunch length in [m].
    """
    # fmt: off
    return (
        circumference
        / (2.0 * np.pi * harmonic_number)
        * np.arccos(1 - (sigma_e**2 * total_energy_GeV * abs(slip_factor) * harmonic_number * np.pi) / (beta_rel**2 * particle_charge * rf_voltage_GV))
    )
    # fmt: on


# This is ionEnergySpread from Michalis's code, in general_functions.py
# The arguments used to be Circumferance, Harmonic_Num, Energy_total, SlipF, BL, beta_rel, RF_Voltage, Energy_loss, Z
def ion_energy_spread(
    circumference: float,
    harmonic_number: int,
    total_energy_GeV: float,
    slip_factor: float,
    bunch_length: float,
    beta_rel: float,
    rf_voltage_GV: float,
    particle_charge: int,
) -> float:
    """
    .. versionadded:: 0.2.0

    Counterpart of the `ion_bunch_length` function, analytically calculates for bunch length for
    protons / electrons. The same caveats than in `ion_bunch_length` apply here.

    Args:
        circumference (float): machine circumference in [m].
        harmonic_number (int): harmonic number of the RF system.
        total_energy_GeV (float): total energy of the simulated particles in [GeV].
        slip_factor (float): slip factor of the machine.
        bunch_length (float): ion bunch length in [m].
        beta_rel (float): relativistic beta of the simulated particles.
        rf_voltage_GV (float): RF voltage of the machine's cavities in [GV].
        particle_charge (int): elementary particle charge, in # of Coulomb charges (for
            instance 1 for electron or proton).

    Returns:
        The analytically calculated dimensionless energy spread for an ion bunch.
    """
    # TODO: check implementation
    tau_phi = 2 * np.pi * harmonic_number * bunch_length / circumference  # bunch length in rad?
    return np.sqrt(
        beta_rel**2
        * particle_charge
        * rf_voltage_GV
        * (-(np.cos(tau_phi) - 1))
        / (total_energy_GeV * abs(slip_factor) * harmonic_number * np.pi)
    )
