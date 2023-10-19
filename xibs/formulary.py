"""
.. _xibs-utils:

Formulary
---------

Module with commonly used formulae to compute quantities of interest needed in the rest of the package.
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

import logging

import numba
import numpy as np

from numpy.typing import ArrayLike

LOGGER = logging.getLogger(__name__)


@numba.njit()
def phi(beta: ArrayLike, alpha: ArrayLike, dx: ArrayLike, dpx: ArrayLike) -> ArrayLike:
    """Phi parameter of Eq (15) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.

    .. todo::
        Figure out exactly who's what here, and why the calculation is not as Eq (15).

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
@numba.njit()
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
    """Analytical calculation for bunch length for protons / electrons (from Wiedermann's book).

    This is a linear approximation which assumes that particles are in the center of the bucket only / mostly.

    .. todo::
        Figure out exactly the formula that was implemented here and reference / document it in the docstring.

    Args:
        circumference (float): machine circumference in [m].
        harmonic_number (int): harmonic number of the RF system.
        total_energy_GeV (float): total energy of the simulated particles in [GeV].
        slip_factor (float): slip factor of the machine.
        sigma_e (float): energy spread of the particles. TODO: check with Michail and in Wiedermann book.
        beta_rel (float): relativistic beta of the simulated particles.
        rf_voltage_GV (float): RF voltage of the machine's cavities in [GV].
        energy_loss_GeV (float): The turn-by-turn oarticle energy loss in [GeV].
        particle_charge (int): elementary particle charge, in # of Coulomb charges (for instance 1 for electron or proton).

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


# This is EnergySpread from Michail's code, in general_functions.py
# The arguments used to be Circumferance, Harmonic_Num, Energy_total, SlipF, BL, beta_rel, RF_Voltage, Energy_loss, Z
@numba.njit()
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
    """Counterpart of the `bunch_length` function, analytically calculates for bunch length for protons / electrons (from Wiedermann's book).

    The same caveats than in `bunch_length` apply here.

    Args:
        circumference (float): machine circumference in [m].
        harmonic_number (int): harmonic number of the RF system.
        total_energy_GeV (float): total energy of the simulated particles in [GeV].
        slip_factor (float): slip factor of the machine.
        bunch_length (float): bunch length in [m].
        beta_rel (float): relativistic beta of the simulated particles.
        rf_voltage_GV (float): RF voltage of the machine's cavities in [GV].
        energy_loss_GeV (float): The turn-by-turn oarticle energy loss in [GeV].
        particle_charge (int): elementary particle charge, in # of Coulomb charges (for instance 1 for electron or proton).

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


# This is ion_BunchLength from Michail's code, in general_functions.py
# The arguments used to be Circumferance, Harmonic_Num, Energy_total, SlipF, Sigma_E, beta_rel, RF_Voltage, Z
@numba.njit()
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
    """Analytical calculation for bunch length for ions (from Wiedermann's book).

    This calculation does not work too well if the bucket is full. Implementation so far as I
    can tell is as used in the scripts for LEIR for ions (some MAD-X script that LEIR studies were using for IBS).

    .. todo::
        Figure out exactly the formula that was implemented here and reference / document it in the docstring.

    Args:
        circumference (float): machine circumference in [m].
        harmonic_number (int): harmonic number of the RF system.
        total_energy_GeV (float): total energy of the simulated particles in [GeV].
        slip_factor (float): slip factor of the machine.
        sigma_e (float): energy spread of the particles? TODO: check with Michail and in Wiedermann book.
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


# This is ionEnergySpread from Michail's code, in general_functions.py
# The arguments used to be Circumferance, Harmonic_Num, Energy_total, SlipF, BL, beta_rel, RF_Voltage, Energy_loss, Z
@numba.njit()
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
    """Counterpart of the `ion_bunch_length` function, analytically calculates for bunch length for protons / electrons.

    The same caveats than in `ion_bunch_length` apply here.

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


@numba.njit()
def iterative_RD(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
    r"""Computes the terms inside the elliptic integral in Eq (4) of
    :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.

    This is an iterative method implementation that was found by Michail (in ``C++``
    then adapted). The implementation is found in ref [5] (uses ref [4] too) of the
    same paper: :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.

    .. note::
        This calculation is taken from the `NagaitsevIBS.RDiter` method in Michalis's
        old code. Some PowerPoints from him in an old ABP group meeting mention how
        the calculation works. One can look into this for details and "documentation".

    Args:
        x (ArrayLike): the :math:`\lambda_1` values in Nagaitsev paper? Eigen values of
            the :math:`\bf{A}` matrix in Eq (2) which comes from B&M (ref ?). In B&M
            it is :math:`\bf{L}` matrix (ref?). This is an array with the value for each
            element in the lattice.
        y (ArrayLike): the :math:`\lambda_2` values in Nagaitsev paper? Eigen values of
            the :math:`\bf{A}` matrix in Eq (2) which comes from B&M (ref ?). In B&M
            it is :math:`\bf{L}` matrix (ref?). This is an array with the value for each
            element in the lattice.
        z (ArrayLike): the :math:`\lambda_3` values in Nagaitsev paper? Eigen values of
            the :math:`\bf{A}` matrix in Eq (2) which comes from B&M (ref ?). In B&M
            it is :math:`\bf{L}` matrix (ref?). This is an array with the value for each
            element in the lattice.

    Returns:
        An array with the result of the calculation for each element in the lattice. This
        is NOT the elliptic integral yet, it has to be integrated afterwards.
    """
    R = []
    for i, j, k in zip(x, y, z):
        x0 = i
        y0 = j
        z0 = k
        if (x0 < 0) and (y0 <= 0) and (z0 <= 0):
            return
        x = x0
        y = y0
        z = [z0]
        li = []
        Sn = []
        differ = 10e-4
        for n in range(0, 1000):
            xi = x
            yi = y
            li.append(np.sqrt(xi * yi) + np.sqrt(xi * z[n]) + np.sqrt(yi * z[n]))
            x = (xi + li[n]) / 4.0
            y = (yi + li[n]) / 4.0
            z.append((z[n] + li[n]) / 4.0)
            if (
                (abs(x - xi) / x0 < differ)
                and (abs(y - yi) / y0 < differ)
                and (abs(z[n] - z[n + 1]) / z0 < differ)
            ):
                break
        lim = n
        mi = (xi + yi + 3 * z[lim]) / 5.0
        Cx = 1 - (xi / mi)
        Cy = 1 - (yi / mi)
        Cz = 1 - (z[n] / mi)
        En = max(Cx, Cy, Cz)
        if En >= 1:
            return
        summ = 0
        for m in range(2, 6):
            Sn.append((Cx**m + Cy**m + 3 * Cz**m) / (2 * m))
        for m in range(0, lim):
            summ += 1 / (np.sqrt(z[m]) * (z[m] + li[m]) * 4**m)

        # Ern = 3 * En**6 / (1 - En) ** (3 / 2.0)
        rn = -Sn[2 - 2] ** 3 / 10.0 + 3 * Sn[3 - 2] ** 2 / 10.0 + 3 * Sn[2 - 2] * Sn[4 - 2] / 5.0
        R.append(
            3 * summ
            + (
                1
                + 3 * Sn[2 - 2] / 7.0
                + Sn[3 - 2] / 3.0
                + 3 * Sn[2 - 2] ** 2 / 22.0
                + 3 * Sn[4 - 2] / 11.0
                + 3 * Sn[2 - 2] * Sn[3 - 2] / 13.0
                + 3 * Sn[5 - 2] / 13.0
                + rn
            )
            / (4**lim * mi ** (3 / 2.0))
        )
    # This returns an array with one value per element in the lattice
    # This is NOT the elliptic integral yet, it has to be integrated afterwards. It is the term in the integral in Eq (4) in Nagaitsev paper.
    return R
