"""
.. _xibs-nagaitsev:

Nagaitsev Integrals
-------------------

Module with functionality to perform analytical calculations according to Nagaitsev's formalism.
A user-facing class is provided which computes the Nagaitsev integrals based on beam optics.
The formalism from which formulas and calculations are implemented can be found in :cite:p:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

import logging

from dataclasses import InitVar, dataclass, field

import numpy as np

from numpy.typing import ArrayLike
from scipy.constants import c, hbar
from scipy.integrate import quad
from scipy.interpolate import interp1d

LOGGER = logging.getLogger(__name__)

# ----- Dataclasses for inputs to Nagaitsev class ----- #


@dataclass
class BeamParameters:
    """Container dataclass for necessary beam parameters. It is initiated from
    the `xpart.Particles` object to track in your line with ``xsuite``.

    Args:
        particles (xpart.Particles): the generated particles to be tracked or used
            in the line.

    Attributes:
        n_part (int): number of simulated particles.
        particle_charge (int): elementary particle charge, in # of Coulomb charges (for
            instance 1 for electron or proton).
        particle_mass_GeV (float): particle mass in [GeV].
        total_energy_GeV (float): total energy of the simulated particles in [GeV].
        gamma_rel (float): relativistic gamma of the simulated particles.
        beta_rel (float): relativistic beta of the simulated particles.
        particle_classical_radius_m (float): the particles' classical radius in [m].
    """

    # ----- To be provided at initialization ----- #
    particles: InitVar["xpart.Particles"]  # Almost all is derived from there, this is not kept!
    # ----- Below are attributes derived from the Particles object ----- #
    # The following are Npart, Ncharg, E_rest, EnTot, gammar, betar and c_rad in Michail's code
    n_part: int = field(init=False)
    particle_charge: int = field(init=False)
    particle_mass_GeV: float = field(init=False)
    total_energy_GeV: float = field(init=False)
    gamma_rel: float = field(init=False)
    beta_rel: float = field(init=False)
    particle_classical_radius_m: float = field(init=False)

    def __post_init__(self, particles: "xpart.Particles"):
        # Attributes derived from the Particles object
        LOGGER.debug("Initializing BeamParameters from Particles object")
        self.n_part = particles.weight[0] * particles.gamma0.shape[0]
        self.particle_charge = particles.q0
        self.particle_mass_GeV = particles.mass0 * 1e-9
        self.total_energy_GeV = np.sqrt(particles.p0c[0] ** 2 + particles.mass0**2) * 1e-9
        self.gamma_rel = particles.gamma0[0]
        self.beta_rel = particles.beta0[0]
        self.particle_classical_radius_m = particles.get_classical_particle_radius0()


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
    twiss: InitVar["xtrack.twiss.TwissTable"]  # Almost all is derived from there, this is not kept!
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
        LOGGER.debug("Initializing OpticsParameters from TwissTable object")
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
        LOGGER.debug("Interpolating beta and dispersion functions")
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
    """
    A single class to compute Nagaitsev integrals (see
    :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`)
    and IBS growth rates. It initiates from a `BeamParameters` and an `OpticsParameters` objects.
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        self.beam_parameters = beam_params
        self.optics = optics

    def coulomb_log(
        self, geom_epsx: float, geom_epxy: float, sigma_delta: float, bunch_length: float
    ) -> float:
        r"""
        Calculates the Coulomb logarithm based on the beam parameters and optics the class
        was initiated with. For a good introductory resource on the Coulomb Log, see:
        https://docs.plasmapy.org/en/stable/notebooks/formulary/coulomb.html

        .. note::
            This is a copy-paste of the body from `CoulogConst` in Michail's code,
            but stopping and returning as soon as the Coulomb logarithm is calculated.
            The caller function is left to compute the rest of the constant term in
            Eq (9) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.

        .. todo::
            Figure out what the hell is going on in this function and document it properly.
            Michail thinks he might have gotten it from the ``MAD-X`` ``IBS`` module?

        Args:
            epsx (float): horizontal geometric emittance in [m].
            epxy (float): vertical geometric emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): bunch length in [m].

        Returns:
            The dimensionless Coulomb logarithm :math:`\ln \left( Λ \right)`.
        """
        LOGGER.debug("Computing Coulomb logarithm for definded beam and optics parameters")
        # fmt: off
        Etrans = (  # who the fuck are you?
            5e8 * (self.beam_parameters.gamma_rel * self.beam_parameters.total_energy_GeV - self.beam_parameters.particle_mass_GeV) * (geom_epsx / self.optics._bx_bar)
        )
        # fmt: on
        TempeV = 2.0 * Etrans

        # TODO: computing sigmas, why?
        sigma_x_cm = 100 * np.sqrt(geom_epsx * self.optics._bx_bar + (self.optics._dx_bar * sigma_delta) ** 2)
        sigma_y_cm = 100 * np.sqrt(geom_epxy * self.optics._by_bar + (self.optics._dy_bar * sigma_delta) ** 2)
        sigma_t_cm = 100 * bunch_length

        # TODO: computing volume, density and maybe Debye length?
        volume = 8.0 * np.sqrt(np.pi**3) * sigma_x_cm * sigma_y_cm * sigma_t_cm
        density = self.beam_parameters.n_part / volume
        debyul = 743.4 * np.sqrt(TempeV / density) / self.beam_parameters.particle_charge  # Debye length?

        rmincl = 1.44e-7 * self.beam_parameters.particle_charge**2 / TempeV
        rminqm = hbar * c * 1e5 / (2.0 * np.sqrt(2e-3 * Etrans * self.beam_parameters.particle_mass_GeV))

        # Now compute the impact parameters and Coulomb logarithm
        bmin = max(rmincl, rminqm)
        bmax = min(sigma_x_cm, debyul)
        return np.log(bmax / bmin)

    def coulomb_log_full_constant(
        self, geom_epsx: float, geom_epxy: float, sigma_delta: float, bunch_length: float
    ) -> float:
        """
        This is the full constant factor (building on Coulomb log, see the `coulomb_log` method)
        from Eq (9) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`. This returns
        the same as what is returned by the `CoulogConst` method in Michail's code.

        .. todo::
            Check the calculation for the rest of the constant term, the expression
            in Eq (9) seems to have been massaged quite a bit.

        Args:
            epsx (float): horizontal geometric emittance in [m].
            epxy (float): vertical geometric emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): bunch length in [m].

        Returns:
            The dimensionless Coulomb logarithm :math:`\ln\\left(Λ\\right)` multiplied by the rest
            of the constant term in Eq (9) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.
        """
        # First get the coulomb logarithm from the dedicated method
        coulomb_logarithm = self.coulomb_log(geom_epsx, geom_epxy, sigma_delta, bunch_length)
        # Then the rest of the constant term in the equation
        # fmt: off
        Ncon = (
            self.beam_parameters.n_part
            * self.beam_parameters.particle_classical_radius_m**2
            * c 
            / (12 * np.pi * self.beam_parameters.beta_rel**3 * self.beam_parameters.gamma_rel**5 * bunch_length)
        )
        # fmt: on
        return Ncon * coulomb_logarithm

    def RDiter(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        r"""Computes the terms inside the elliptic integral in Eq (4) of
        :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.

        This is an iterative method implementation that was found by Michail (in ``C++``
        then adapted). The implementation is found in ref [5] (uses ref [4] too) of the
        same paper: :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.

        .. note::
            This is for now a copy-paste of the `RDiter` method in Michail's code.
            Some PowerPoints from Michail in an old ABP group meeting mention how this
            calculation works. Can look into this for details and documentation.

        .. todo::
            This is the most time-consuming part of the class's integrals computing. For
            optimization, since this doesn't call any internal attributes of the class
            it could be moved out (into `xibs.formulary`?) and potentially JIT-compiled with
            ``numba``. Then we import directly from the right module and call it in the
            integrals calculations.

        .. todo::
            Go through the old scripts in debugging mode and inspect what is passed in
            and then out to get a better idea of the function signature.

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
        LOGGER.debug("Iteratively computing elliptic integral RD term")
        R = []
        for i, j, k in zip(x, y, z):
            x0 = i
            y0 = j
            z0 = k
            if (x0 < 0) and (y0 <= 0) and (z0 <= 0):
                print("Elliptic Integral Calculation Failed. Wrong input values!")
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
                print("Something went wrong with En")
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
