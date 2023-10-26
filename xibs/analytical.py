"""
.. _xibs-analytical:

Analytical Calculations
-----------------------

Module with functionality to perform analytical calculations according to Nagaitsev's formalism.
A user-facing class is provided which computes the Nagaitsev integrals based on beam parameters and machine optics.
The formalism from which formulas and calculations are implemented can be found in :cite:p:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

import warnings

from dataclasses import astuple, dataclass
from logging import getLogger
from typing import Tuple

import numpy as np

from scipy.constants import c, hbar
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import elliprd

from xibs.formulary import phi
from xibs.inputs import BeamParameters, OpticsParameters

LOGGER = getLogger(__name__)
warnings.filterwarnings("ignore")  # scipy integration routines might warn for subdivisions

# ----- Dataclasses to store results ----- #


@dataclass
class NagaitsevIntegrals:
    """
    .. versionadded:: 0.2.0

    Container dataclass for Nagaitsev integrals results.

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
    """
    .. versionadded:: 0.2.0

    Container dataclass for IBS growth rates results.

    Args:
        Tx (float): horizontal IBS growth rate.
        Ty (float): vertical IBS growth rate.
        Tz (float): longitudinal IBS growth rate.
    """

    Tx: float
    Ty: float
    Tz: float


# ----- Main class to compute Nagaitsev integrals and IBS growth rates ----- #


class NagaitsevIBS:
    """
    .. versionadded:: 0.2.0

    A single class to compute Nagaitsev integrals (see
    :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`)
    and IBS growth rates. It initiates from a `BeamParameters` and an `OpticsParameters` objects.

    Attributes:
        beam_parameters (BeamParameters): the beam parameters to use for the calculations.
        optics (OpticsParameters): the optics parameters to use for the calculations.
        elliptic_integrals (NagaitsevIntegrals): the computed elliptic integrals. This
            self-updates when they are computed with the `integrals` method.
        ibs_growth_rates (IBSGrowthRates): the computed IBS growth rates. This self-updates
            when they are computed with the `growth_rates` method.
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        self.beam_parameters: BeamParameters = beam_params
        self.optics: OpticsParameters = optics
        # These self-update when they are computed, but can be overwritten by the user
        self.elliptic_integrals: NagaitsevIntegrals = None
        self.ibs_growth_rates: IBSGrowthRates = None

    def coulomb_log(
        self, geom_epsx: float, geom_epxy: float, sigma_delta: float, bunch_length: float
    ) -> float:
        r"""
        .. versionadded:: 0.2.0

        Calculates the Coulomb logarithm based on the beam parameters and optics the class
        was initiated with. For a good introductory resource on the Coulomb Log, see:
        https://docs.plasmapy.org/en/stable/notebooks/formulary/coulomb.html

        .. note::
            This function follows the exact computing implementation of the Coulomb log
            calculation in the ``MAD-X`` source code. One can find it in the source code
            in the file `MAD-X/src/ibsdb.f90` as the `twclog` subroutine.

        Args:
            epsx (float): horizontal geometric emittance in [m].
            epxy (float): vertical geometric emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): bunch length in [m].

        Returns:
            The dimensionless Coulomb logarithm :math:`\ln \left( \Lambda \right)`.
        """
        LOGGER.debug("Computing Coulomb logarithm for definded beam and optics parameters")
        # ----------------------------------------------------------------------------------------------
        # Interpolated beta and dispersion functions for the average calculation below
        LOGGER.debug("Interpolating beta and dispersion functions")
        _bxb = interp1d(self.optics.s, self.optics.betx)
        _byb = interp1d(self.optics.s, self.optics.bety)
        _dxb = interp1d(self.optics.s, self.optics.dx)
        _dyb = interp1d(self.optics.s, self.optics.dy)
        # ----------------------------------------------------------------------------------------------
        # Computing "average" of these functions - better here than a simple np.mean
        # calculation because the latter doesn't take in consideration element lengths
        # and can be skewed by some very high peaks in the optics
        _bx_bar = quad(_bxb, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference
        _by_bar = quad(_byb, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference
        _dx_bar = quad(_dxb, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference
        _dy_bar = quad(_dyb, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference
        # ----------------------------------------------------------------------------------------------
        # Calculate transverse temperature as 2*P*X, i.e. assume the transverse energy is temperature/2
        # fmt: off
        Etrans = (  
            5e8
            * (self.beam_parameters.gamma_rel * self.beam_parameters.total_energy_GeV - self.beam_parameters.particle_mass_GeV)
            * (geom_epsx / _bx_bar)
        )
        # fmt: on
        TempeV = 2.0 * Etrans
        # ----------------------------------------------------------------------------------------------
        # Compute sigmas in each dimension
        sigma_x_cm = 100 * np.sqrt(geom_epsx * _bx_bar + (_dx_bar * sigma_delta) ** 2)
        sigma_y_cm = 100 * np.sqrt(geom_epxy * _by_bar + (_dy_bar * sigma_delta) ** 2)
        sigma_t_cm = 100 * bunch_length
        # ----------------------------------------------------------------------------------------------
        # Calculate beam volume to get density (in cm^{-3}) then Debye length
        volume = 8.0 * np.sqrt(np.pi**3) * sigma_x_cm * sigma_y_cm * sigma_t_cm
        density = self.beam_parameters.n_part / volume
        debyul = 743.4 * np.sqrt(TempeV / density) / self.beam_parameters.particle_charge  # Debye length?
        # ----------------------------------------------------------------------------------------------
        # Calculate 'rmin' as larger of classical distance of closest approach or quantum mechanical
        # diffraction limit from nuclear radius
        rmincl = 1.44e-7 * self.beam_parameters.particle_charge**2 / TempeV
        rminqm = hbar * c * 1e5 / (2.0 * np.sqrt(2e-3 * Etrans * self.beam_parameters.particle_mass_GeV))
        # ----------------------------------------------------------------------------------------------
        # Now compute the impact parameters and finally Coulomb logarithm
        bmin = max(rmincl, rminqm)
        bmax = min(sigma_x_cm, debyul)
        return np.log(bmax / bmin)

    # This is 'Nagaitsev_Integrals' from Michalis's old code but it stops a bit earlier and really returns the integrals
    # The arguments used to be named Emit_x, Emit_y, Sig_M, BunchL there
    def integrals(self, geom_epsx: float, geom_epsy: float, sigma_delta: float) -> NagaitsevIntegrals:
        r"""
        .. versionadded:: 0.2.0

        Computes the Nagaitsev integrals, named :math:`I_x, I_y` and :math:`I_z` in this code base.

        These correspond to the integrals inside of Eq (32), (31) and (30) in
        :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`, respectively.
        The instance attribute `self.elliptic_integrals` is automatically updated
        with the results of this method. It is used for other calculations.

        .. tip::
            The calculation is done according to the following steps, which are related to different
            equations in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`:

                - Computes various intermediate terms and then :math:`a_x, a_y, a_s, a_1` and :math:`a_2` constants from Eq (18-21).
                - Computes the eigenvalues :math:`\lambda_1, \lambda_2` of the :math:`\bf{A}` matrix (:math:`\bf{L}` matrix in B&M) from Eq (22-24).
                - Iteratively computes the :math:`R_1, R_2` and :math:`R_3` terms from Eq (25-27) with the forms of Eq (5-6).
                - Computes the :math:`S_p, S_x` and :math:`S_{xp}` terms from Eq (33-35).
                - Computes and return the integrals terms in Eq (30-32).

        Args:
            epsx (float): horizontal geometric emittance in [m].
            epxy (float): vertical geometric emittance in [m].
            sigma_delta (float): momentum spread.

        Returns:
            A `NagaitsevIntegrals` object with the computed integrals for each plane.
        """
        LOGGER.info("Computing Nagaitsev integrals for defined beam and optics parameters")
        # fmt: off
        # All of the following (when type annotated as np.ndarray), hold one value per element in the lattice
        # ----------------------------------------------------------------------------------------------
        # Computing necessary intermediate terms for the following lines
        sigx: np.ndarray = np.sqrt(self.optics.betx * geom_epsx + (self.optics.dx * sigma_delta) ** 2)
        sigy: np.ndarray = np.sqrt(self.optics.bety * geom_epsy + (self.optics.dy * sigma_delta) ** 2)
        phix: np.ndarray = phi(self.optics.betx, self.optics.alfx, self.optics.dx, self.optics.dpx)
        # Computing the constants from Eq (18-21) in Nagaitsev paper
        ax: np.ndarray = self.optics.betx / geom_epsx
        ay: np.ndarray = self.optics.bety / geom_epsy
        a_s: np.ndarray = ax * (self.optics.dx**2 / self.optics.betx**2 + phix**2) + 1 / sigma_delta**2
        a1: np.ndarray = (ax + self.beam_parameters.gamma_rel**2 * a_s) / 2.0
        a2: np.ndarray = (ax - self.beam_parameters.gamma_rel**2 * a_s) / 2.0
        sqrt_term = np.sqrt(a2**2 + self.beam_parameters.gamma_rel**2 * ax**2 * phix**2)  # square root term in Eq (22-23) and Eq (33-35)
        # ----------------------------------------------------------------------------------------------
        # These are from Eq (22-24) in Nagaitsev paper, eigen values of A matrix (L matrix in B&M)
        lambda_1: np.ndarray = ay
        lambda_2: np.ndarray = a1 + sqrt_term
        lambda_3: np.ndarray = a1 - sqrt_term
        # ----------------------------------------------------------------------------------------------
        # These are the R_D terms to compute, from Eq (25-27) in Nagaitsev paper (at each element of the lattice)
        LOGGER.debug("Computing elliptic integrals R1, R2 and R3")
        R1: np.ndarray = elliprd(1 / lambda_2, 1 / lambda_3, 1 / lambda_1) / lambda_1
        R2: np.ndarray = elliprd(1 / lambda_3, 1 / lambda_1, 1 / lambda_2) / lambda_2
        R3: np.ndarray = 3 * np.sqrt(lambda_1 * lambda_2 / lambda_3) - lambda_1 * R1 / lambda_3 - lambda_2 * R2 / lambda_3
        # ----------------------------------------------------------------------------------------------
        # This are the terms from Eq (33-35) in Nagaitsev paper
        Sp: np.ndarray = (2 * R1 - R2 * (1 - 3 * a2 / sqrt_term) - R3 * (1 + 3 * a2 / sqrt_term)) * 0.5 * self.beam_parameters.gamma_rel**2
        Sx: np.ndarray = (2 * R1 - R2 * (1 + 3 * a2 / sqrt_term) - R3 * (1 - 3 * a2 / sqrt_term)) * 0.5
        Sxp: np.ndarray = 3 * self.beam_parameters.gamma_rel**2 * phix**2 * ax * (R3 - R2) / sqrt_term
        # ----------------------------------------------------------------------------------------------
        # These are the integrands of the integrals in Eq (30-32) in Nagaitsev paper
        Ix_integrand = (
            self.optics.betx
            / (self.optics.circumference * sigx * sigy)
            * (Sx + Sp * (self.optics.dx**2 / self.optics.betx**2 + phix**2) + Sxp)
        )
        Iy_integrand = self.optics.bety / (self.optics.circumference * sigx * sigy) * (R2 + R3 - 2 * R1)
        Ip_integrand = Sp / (self.optics.circumference * sigx * sigy)
        # ----------------------------------------------------------------------------------------------
        # Integrating the integrands above accross the ring to get the desired results
        Ix = float(np.sum(Ix_integrand[:-1] * np.diff(self.optics.s)))
        Iy = float(np.sum(Iy_integrand[:-1] * np.diff(self.optics.s)))
        Iz = float(np.sum(Ip_integrand[:-1] * np.diff(self.optics.s)))
        result = NagaitsevIntegrals(Ix, Iy, Iz)
        # fmt: on
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.elliptic_integrals = result
        return result

    # This is the end of the calculations in 'Nagaitsev_Integrals' from Michalis's old code (the last 3 lines essentially)
    # The arguments used to be named Emit_x, Emit_y, Sig_M, BunchL there
    def growth_rates(
        self, geom_epsx: float, geom_epsy: float, sigma_delta: float, bunch_length: float
    ) -> IBSGrowthRates:
        r"""
        .. versionadded:: 0.2.0

        Computes the ``IBS`` growth rates, named :math:`T_x, T_y` and :math:`T_z` in this
        code base, from Nagaitsev integrals. These correspond to the :math:`1 / \tau` term,
        for each plane, of Eq (28) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`,
        respectively. The instance attribute `self.ibs_growth_rates` is automatically updated
        with the results of this method.

        .. warning::
            This calculation is done by building on the Nagaitsev integrals. If the
            latter have not been computed yet, this method will raise an error. Please
            remember to call the instance's `integrals` method first.

        .. warning::
            Currently this calculation does not take into account vertical dispersion.
            We are working on implementing it for a future version.

        .. tip::
            The calculation is done according to the following steps, which are related to different
            equations in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`:

                - Get the Nagaitsev integrals from the instance attributes (integrals of Eq (30-32)).
                - Computes the Coulomb logarithm for the defined beam and optics parameters.
                - Compute the rest of the constant term of Eq (30-32).
                - Compute for each plane the full result of Eq (30-32), respectively.
                - Plug these into Eq (28) and divide by either :math:`\varepsilon_x, \varepsilon_y` or :math:`\sigma_{\delta}^{2}` (as relevant) to get :math:`1 / \tau`.

        Args:
            epsx (float): horizontal geometric emittance in [m].
            epxy (float): vertical geometric emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): the bunch length in [m].

        Raises:
            ValueError: if the Nagaitsev integrals have not yet been computed.

        Returns:
            An `IBSGrowthRates` object with the computed growth rates for each plane.
        """
        # ----------------------------------------------------------------------------------------------
        # Check that the Nagaitsev integrals have been computed beforehand
        if self.elliptic_integrals is None:
            LOGGER.error("Attempted to compute growth rates without having computed Nagaitsev integrals.")
            raise ValueError(
                "Nagaitsev integrals have not been computed yet, cannot compute growth rates.\n"
                "Please call the `integrals` method first."
            )
        LOGGER.info("Computing IBS growth rates for defined beam and optics parameters")
        # ----------------------------------------------------------------------------------------------
        # Get the Coulomb logarithm and the rest of the constant term in Eq (30-32)
        coulomb_logarithm = self.coulomb_log(geom_epsx, geom_epsy, sigma_delta, bunch_length)
        # Then the rest of the constant term in the equation
        # fmt: off
        rest_of_constant_term = (
            self.beam_parameters.n_part * self.beam_parameters.particle_classical_radius_m**2 * c 
            / (12 * np.pi * self.beam_parameters.beta_rel**3 * self.beam_parameters.gamma_rel**5 * bunch_length)
        )
        # fmt: on
        full_constant_term = rest_of_constant_term * coulomb_logarithm
        # ----------------------------------------------------------------------------------------------
        # Compute the full result of Eq (30-32) for each plane | make sure to convert back to float
        Ix, Iy, Iz = astuple(self.elliptic_integrals)
        Tx = float(Ix * full_constant_term / geom_epsx)
        Ty = float(Iy * full_constant_term / geom_epsy)
        Tz = float(Iz * full_constant_term / sigma_delta**2)
        result = IBSGrowthRates(Tx, Ty, Tz)
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.ibs_growth_rates = result
        return result

    # This is 'emit_evol' from Michalis's old code
    # The arguments used to be named Emit_x, Emit_y, Sig_M, BunchL (unused) and dt there
    def emittance_evolution(
        self, geom_epsx: float, geom_epsy: float, sigma_delta: float, dt: float = None
    ) -> Tuple[float, float, float]:
        r"""
        .. versionadded:: 0.2.0

        Analytically computes the new emittances after a given time step `dt` has
        ellapsed, from initial values, based on the ``IBS`` growth rates.

        .. warning::
            This calculation is done by building on the ``IBS`` growth rates. If the
            latter have not been computed yet, this method will raise an error. Please
            remember to call the instance's `growth_rates` method first.

        .. tip::
            The calculation is an exponential growth based on the rates :math:`T_{x,y,z}`. It goes
            according to:

            .. math::

                T_{x,y,z} &= 1 / \tau_{x,y,z}

                \varepsilon_{x,y}^{N+1} &= \varepsilon_{x,y}^{N} + e^{t / \tau_{x,y}}

                \sigma_{\delta}^{N+1} &= \sigma_{\delta}^{N} + e^{t / 2 \tau_{z}}


        Args:
            epsx (float): horizontal geometric emittance in [m].
            epxy (float): vertical geometric emittance in [m].
            sigma_delta (float): momentum spread.
            dt (float, optional): the time interval to use. Default to the inverse
                of the revolution frequency, :math:`1 / f_{rev}`.

        Raises:
            ValueError: if the IBS growth rates have not yet been computed.

        Returns:
            A tuple with the new horizontal & vertical geometric emittances as well as the new
            momentum spread, after the time step has ellapsed.
        """
        # ----------------------------------------------------------------------------------------------
        # Check that the IBS growth rates have been computed beforehand
        if self.ibs_growth_rates is None:
            LOGGER.error("Attempted to compute emittance evolution without having computed growth rates.")
            raise ValueError(
                "IBS growth rates have not been computed yet, cannot compute new emittances.\n"
                "Please call the `growth_rates` method first."
            )
        LOGGER.info("Computing new emittances from IBS growth rates for defined beam and optics parameters")
        # ----------------------------------------------------------------------------------------------
        # Set the time step to 1 / frev if not provided
        if dt is None:
            LOGGER.debug("No time step provided, defaulting to 1 / frev")
            dt = 1 / self.optics.revolution_frequency
        # ----------------------------------------------------------------------------------------------
        # Compute new emittances and return them. Here we multiply because T = 1 / tau
        new_epsx = geom_epsx * np.exp(dt * float(self.ibs_growth_rates.Tx))
        new_epsy = geom_epsy * np.exp(dt * float(self.ibs_growth_rates.Ty))
        new_sigma_delta = sigma_delta * np.exp(dt * float(0.5 * self.ibs_growth_rates.Tz))
        return new_epsx, new_epsy, new_sigma_delta
