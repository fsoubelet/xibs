"""
.. _xibs-analytical:

IBS: Analytical Calculations
----------------------------

Module with functionality to perform analytical IBS calculations, either according to either Nagaitsev's or Bjorken & Mtingwa's formalism.
User-facing classes are provided which allow to compute the growth rates based on beam parameters and machine optics.
The formalism from which formulas and calculations are implemented can be found in :cite:p:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation` and :cite:`CERN:Antoniou:Revision_IBS_MADX`, respectively.

.. warning::
    Please note that these analytical implementations make the assumptions.
    Should your scenario not satisfy the following assumptions, the results might not be accurate:

        - It is assumed that beam profiles are Gaussian,
        - It is assumed that no betatron coupling is present in the machine.
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

import warnings

from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass
from logging import getLogger
from typing import Callable, Tuple

import numpy as np

from numpy.typing import ArrayLike
from scipy.constants import c, hbar
from scipy.integrate import quad, quad_vec
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


# ----- Abstract Base Class to Inherit from ----- #


class AnalyticalIBS(ABC):
    r"""
    .. versionadded:: 0.4.0

    Abstract base class for analytical IBS calculations, from which all
    implementations inherit.

    Attributes:
        beam_parameters (BeamParameters): the beam parameters to use for the calculations.
        optics (OpticsParameters): the optics parameters to use for the calculations.
        ibs_growth_rates (IBSGrowthRates): the computed IBS growth rates. This self-updates
            when they are computed with the `growth_rates` method.
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        self.beam_parameters: BeamParameters = beam_params
        self.optics: OpticsParameters = optics
        # This one self-updates when computed, but can be overwritten by the user
        self.ibs_growth_rates: IBSGrowthRates = None

    def __str__(self) -> str:
        has_growth_rates = isinstance(self.ibs_growth_rates, IBSGrowthRates)  # False if default value of None
        return (
            f"{self.__class__.__name__} object for analytical IBS calculations.\n"
            f"IBS growth rates computed: {has_growth_rates}"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def coulomb_log(
        self,
        epsx: float,
        epsy: float,
        sigma_delta: float,
        bunch_length: float,
        bunched: bool = True,
        normalized_emittances: bool = False,
    ) -> float:
        r"""
        .. versionadded:: 0.2.0

        Calculates the Coulomb logarithm based on the beam parameters and optics the class
        was initiated with. For a good introductory resource on the Coulomb Log, see:
        https://docs.plasmapy.org/en/stable/notebooks/formulary/coulomb.html.

        .. note::
            This function follows the formulae in :cite:`AIP:Anderson:Physics_Vade_Mecum`. The
            Coulomb log is computed as :math:`\ln \left( \Lambda \right) = \ln(r_{max} / r_{min})`.
            Here :math:`r_{max}` denotes the smaller of :math:`\sigma_x` and the Debye length; while
            :math:`r_{min}` is the larger of the classical distance of closest approach and the
            quantum diffraction limit from the nuclear radius. It is the calculation that is done by
            ``MAD-X`` (see the `twclog` subroutine in the `MAD-X/src/ibsdb.f90` source file).

        .. note::
            Both geometric or normalized emittances can be given as input to this function, and it is assumed
            the user provides geomettric emittances. If normalized ones are given the `normalized_emittances`
            parameter should be set to `True` (it defaults to `False`). Internally, a conversion is done to
            geometric emittances, which are used in the computations.

        Args:
            epsx (float): horizontal geometric or normalized emittance in [m].
            epsy (float): vertical geometric or normalized emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): bunch length in [m].
            bunched (bool): whether the beam is bunched or not (coasting). Defaults to `True`.
            normalized_emittances (bool): whether the provided emittances are
                normalized or not. Defaults to `False` (assume geometric emittances).

        Returns:
            The dimensionless Coulomb logarithm :math:`\ln \left( \Lambda \right)`.
        """
        LOGGER.debug("Computing Coulomb logarithm for defined beam and optics parameters")
        # ----------------------------------------------------------------------------------------------
        # Make sure we are working with geometric emittances
        geom_epsx = epsx if normalized_emittances is False else self._geometric_emittance(epsx)
        geom_epsy = epsy if normalized_emittances is False else self._geometric_emittance(epsy)
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
        with warnings.catch_warnings():  # Catch and ignore the scipy.integrate.IntegrationWarning
            warnings.simplefilter("ignore", category=UserWarning)
            _bx_bar = quad(_bxb, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference
            _by_bar = quad(_byb, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference
            _dx_bar = quad(_dxb, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference
            _dy_bar = quad(_dyb, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference
        # ----------------------------------------------------------------------------------------------
        # Calculate transverse temperature as 2*P*X, i.e. assume the transverse energy is temperature/2
        # fmt: off
        Etrans = (  
            5e8
            * (self.beam_parameters.gamma_rel
               * self.beam_parameters.total_energy_eV * 1e-9  # total energy needed in GeV
               - self.beam_parameters.particle_mass_eV * 1e-9  # particle mass needed in GeV
            )
            * (geom_epsx / _bx_bar)
        )
        # fmt: on
        TempeV = 2.0 * Etrans
        # ----------------------------------------------------------------------------------------------
        # Compute sigmas in each dimension
        sigma_x_cm = 100 * np.sqrt(geom_epsx * _bx_bar + (_dx_bar * sigma_delta) ** 2)
        sigma_y_cm = 100 * np.sqrt(geom_epsy * _by_bar + (_dy_bar * sigma_delta) ** 2)
        sigma_t_cm = 100 * bunch_length
        # ----------------------------------------------------------------------------------------------
        # Calculate beam volume to get density (in cm^{-3}) then Debye length
        if bunched is True:  # bunched beam
            volume = 8.0 * np.sqrt(np.pi**3) * sigma_x_cm * sigma_y_cm * sigma_t_cm
        else:  # coasting beam
            volume = 4.0 * np.pi * sigma_x_cm * sigma_y_cm * 100 * self.optics.circumference
        density = self.beam_parameters.n_part / volume
        debyul = 743.4 * np.sqrt(TempeV / density) / self.beam_parameters.particle_charge
        # ----------------------------------------------------------------------------------------------
        # Calculate 'rmin' as larger of classical distance of closest approach or quantum mechanical
        # diffraction limit from nuclear radius
        rmincl = 1.44e-7 * self.beam_parameters.particle_charge**2 / TempeV
        rminqm = (
            hbar * c * 1e5 / (2.0 * np.sqrt(2e-3 * Etrans * self.beam_parameters.particle_mass_eV * 1e-9))
        )  # energy in GeV
        # ----------------------------------------------------------------------------------------------
        # Now compute the impact parameters and finally Coulomb logarithm
        bmin = max(rmincl, rminqm)
        bmax = min(sigma_x_cm, debyul)
        return np.log(bmax / bmin)

    @abstractmethod
    def growth_rates(
        self,
        epsx: float,
        epsy: float,
        sigma_delta: float,
        bunch_length: float,
        bunched: bool = True,
        normalized_emittances: bool = False,
    ) -> IBSGrowthRates:
        r"""
        Method to compute the IBS growth rates.

        Args:
            epsx (float): horizontal geometric or normalized emittance in [m].
            epsy (float): vertical geometric or normalized emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): the bunch length in [m].
            bunched (bool): whether the beam is bunched or not (coasting). Defaults to `True`.
            normalized_emittances (bool): whether the provided emittances are
                normalized or not. Defaults to `False` (assume geometric emittances).

        Returns:
            An `IBSGrowthRates` object with the computed growth rates for each plane.
        """
        raise NotImplementedError(
            "This method should be implemented in all child classes, but it hasn't been for this one."
        )

    def emittance_evolution(
        self,
        epsx: float,
        epsy: float,
        sigma_delta: float,
        bunch_length: float,
        dt: float = None,
        normalized_emittances: bool = False,
    ) -> Tuple[float, float, float, float]:
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
            according to the following, where :math:`N` represents the time step:

            .. math::

                T_{x,y,z} &= 1 / \tau_{x,y,z}

                \varepsilon_{x,y}^{N+1} &= \varepsilon_{x,y}^{N} * e^{t / \tau_{x,y}}

                \sigma_{\delta, z}^{N+1} &= \sigma_{\delta, z}^{N} * e^{t / 2 \tau_{z}}

        .. note::
            Both geometric or normalized emittances can be given as input to this function, and it is assumed
            the user provides geomettric emittances. If normalized ones are given the `normalized_emittances`
            parameter should be set to `True` (it defaults to `False`). Internally, a conversion is done to
            geometric emittances, which are used in the computations. The returned emittances correspond to
            the type of those provided: if given normalized emittances this function will return values that
            correspond to the new normalized emittances.

        Args:
            epsx (float): horizontal geometric or normalized emittance in [m].
            epsy (float): vertical geometric or normalized emittance in [m].
            sigma_delta (float): momentum spread.
            dt (float, optional): the time interval to use, in [s]. Default to the inverse
                of the revolution frequency, :math:`1 / f_{rev}`.
            bunch_length (float): the bunch length in [m].
            normalized_emittances (bool): whether the provided emittances are
                normalized or not. Defaults to `False` (assume geometric emittances).

        Raises:
            ValueError: if the ``IBS`` growth rates have not yet been computed.

        Returns:
            A tuple with the new horizontal & vertical geometric emittances, the new
            momentum spread and the new bunch length, after the time step has ellapsed.
        """
        # ----------------------------------------------------------------------------------------------
        # Make sure we are working with geometric emittances
        geom_epsx = epsx if normalized_emittances is False else self._geometric_emittance(epsx)
        geom_epsy = epsy if normalized_emittances is False else self._geometric_emittance(epsy)
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
        new_epsx: float = geom_epsx * np.exp(dt * float(self.ibs_growth_rates.Tx))
        new_epsy: float = geom_epsy * np.exp(dt * float(self.ibs_growth_rates.Ty))
        new_sigma_delta: float = sigma_delta * np.exp(dt * float(0.5 * self.ibs_growth_rates.Tz))
        new_bunch_length: float = bunch_length * np.exp(dt * float(0.5 * self.ibs_growth_rates.Tz))
        # ----------------------------------------------------------------------------------------------
        # Make sure we return the same type of emittances as the user provided
        new_epsx = new_epsx if normalized_emittances is False else self._normalized_emittance(new_epsx)
        new_epsy = new_epsy if normalized_emittances is False else self._normalized_emittance(new_epsy)
        return new_epsx, new_epsy, new_sigma_delta, new_bunch_length

    def _normalized_emittance(self, geometric_emittance: float) -> float:
        r"""
        .. versionadded:: 0.4.0

        Computes normalized emittance from the geometric one, using relativistic
        beta and gamma from the the instance's beam parameters attribute.

        Args:
            geometric_emittance (float): geometric emittance in [m].
            beta_ref (float): relativistic beta.
            gamma_rel (float): relativistic gamma.

        Returns:
            The normalized emittance in [m].
        """
        return geometric_emittance * self.beam_parameters.beta_rel * self.beam_parameters.gamma_rel

    def _geometric_emittance(self, normalized_emittance: float) -> float:
        r"""
        .. versionadded:: 0.4.0

        Computes geometric emittance from the normalized one, using relativistic
        beta and gamma from the the instance's beam parameters attribute.

        Args:
            normalized_emittance (float): normalized emittance in [m].
            beta_ref (float): relativistic beta.
            gamma_rel (float): relativistic gamma.

        Returns:
            The geometric emittance in [m].
        """
        return normalized_emittance / (self.beam_parameters.beta_rel * self.beam_parameters.gamma_rel)


# ----- Classes to Compute Analytical IBS Growth Rates ----- #


class NagaitsevIBS(AnalyticalIBS):
    r"""
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
        super().__init__(beam_params, optics)
        # This self-updates when computed, but can be overwritten by the user
        self.elliptic_integrals: NagaitsevIntegrals = None

    def integrals(
        self, epsx: float, epsy: float, sigma_delta: float, normalized_emittances: bool = False
    ) -> NagaitsevIntegrals:
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
                - Computes the :math:`R_1, R_2` and :math:`R_3` terms from Eq (25-27) with the forms of Eq (5-6).
                - Computes the :math:`S_p, S_x` and :math:`S_{xp}` terms from Eq (33-35).
                - Computes and returns the integrals terms in Eq (30-32).

        .. note::
            Both geometric or normalized emittances can be given as input to this function, and it is assumed
            the user provides geomettric emittances. If normalized ones are given the `normalized_emittances`
            parameter should be set to `True` (it defaults to `False`). Internally, a conversion is done to
            geometric emittances, which are used in the computations.

        Args:
            epsx (float): horizontal geometric or normalized emittance in [m].
            epsy (float): vertical geometric or normalized emittance in [m].
            sigma_delta (float): momentum spread. Defaults to `None`.
            normalized_emittances (bool): whether the provided emittances are
                normalized or not. Defaults to `False` (assume geometric emittances).

        Returns:
            A `NagaitsevIntegrals` object with the computed integrals for each plane.
        """
        LOGGER.info("Computing Nagaitsev integrals for defined beam and optics parameters")
        # fmt: off
        # All of the following (when type annotated as np.ndarray), hold one value per element in the lattice
        # ----------------------------------------------------------------------------------------------
        # Make sure we are working with geometric emittances
        geom_epsx = epsx if normalized_emittances is False else self._geometric_emittance(epsx)
        geom_epsy = epsy if normalized_emittances is False else self._geometric_emittance(epsy)
        # ----------------------------------------------------------------------------------------------
        # Computing necessary intermediate terms for the following lines
        sigx: np.ndarray = np.sqrt(self.optics.betx * geom_epsx + (self.optics.dx * sigma_delta)**2)
        sigy: np.ndarray = np.sqrt(self.optics.bety * geom_epsy + (self.optics.dy * sigma_delta)**2)
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
        Iz_integrand = Sp / (self.optics.circumference * sigx * sigy)
        # ----------------------------------------------------------------------------------------------
        # Integrating the integrands above accross the ring to get the desired results
        # This is identical to np.trapz(Ixyz_integrand, self.optics.s) but faster and somehow closer to MAD-X values
        Ix = float(np.sum(Ix_integrand[:-1] * np.diff(self.optics.s)))
        Iy = float(np.sum(Iy_integrand[:-1] * np.diff(self.optics.s)))
        Iz = float(np.sum(Iz_integrand[:-1] * np.diff(self.optics.s)))
        result = NagaitsevIntegrals(Ix, Iy, Iz)
        # fmt: on
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.elliptic_integrals = result
        return result

    def growth_rates(
        self,
        epsx: float,
        epsy: float,
        sigma_delta: float,
        bunch_length: float,
        bunched: bool = True,
        normalized_emittances: bool = False,
        compute_integrals: bool = True,
    ) -> IBSGrowthRates:
        r"""
        .. versionadded:: 0.2.0

        Computes the ``IBS`` growth rates, named :math:`T_x, T_y` and :math:`T_z` in this
        code base, from Nagaitsev integrals. These correspond to the :math:`1 / \tau` term,
        for each plane, of Eq (28) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`,
        respectively. The instance attribute `self.ibs_growth_rates` is automatically updated
        with the results of this method when it is called.

        .. note::
            This calculation is done by building on the Nagaitsev integrals. If the
            latter have not been computed yet, this method will first log a message
            and compute them, then compute the growth rates.

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

        .. note::
            Both geometric or normalized emittances can be given as input to this function, and it is assumed
            the user provides geomettric emittances. If normalized ones are given the `normalized_emittances`
            parameter should be set to `True` (it defaults to `False`). Internally, a conversion is done to
            geometric emittances, which are used in the computations.

        Args:
            epsx (float): horizontal geometric or normalized emittance in [m].
            epsy (float): vertical geometric or normalized emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): the bunch length in [m].
            bunched (bool): UNIMPLEMENTED AT THE MOMENT. Whether the beam is bunched or not (coasting).
                Defaults to `True`.
            normalized_emittances (bool): whether the provided emittances are
                normalized or not. Defaults to `False` (assume geometric emittances).
            compute_integrals (bool): if `True`, the Nagaitsev elliptic integrals will be computed
                before the growth rates. Defaults to `True`. New in version 0.3.0.

        Returns:
            An `IBSGrowthRates` object with the computed growth rates for each plane.
        """
        # ----------------------------------------------------------------------------------------------
        # Catch and raise an error if the user asks for coasting beam (not implemented yet)
        if bunched is False:
            LOGGER.error(
                "Computing growth rates for coasting beams is currently not supported in this class."
            )
            raise NotImplementedError(
                "Calculation for coasting beams is not implemented yet in this formalism."
                "Please use the BjorkenMtignwaIBS class instead, which supports this feature."
            )
        # ----------------------------------------------------------------------------------------------
        # Make sure we are working with geometric emittances
        geom_epsx = epsx if normalized_emittances is False else self._geometric_emittance(epsx)
        geom_epsy = epsy if normalized_emittances is False else self._geometric_emittance(epsy)
        # ----------------------------------------------------------------------------------------------
        # Check that the Nagaitsev integrals have been computed beforehand
        if self.elliptic_integrals is None and compute_integrals is False:
            LOGGER.info(
                "Computing growth rates requires having computed Nagaitsev integrals. They will be computed first."
            )
            _ = self.integrals(geom_epsx, geom_epsy, sigma_delta)
        # ----------------------------------------------------------------------------------------------
        # Compute the integrals if asked to by the user (default behaviour)
        if compute_integrals is True:
            _ = self.integrals(geom_epsx, geom_epsy, sigma_delta)
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


class BjorkenMtingwaIBS(AnalyticalIBS):
    r"""
    .. versionadded:: 0.3.0

    A single class to compute the IBS growth rates according to the `Bjorken & Mtingwa` formalism.
    The exact approach follows the ``MAD-X`` implementation, which has corrected B&M in order to
    take in consideration the vertical dispersion values (see the relevant note about the changes
    at :cite:`CERN:Antoniou:Revision_IBS_MADX`). It initiates from a `BeamParameters` and an
    `OpticsParameters` objects.

    .. note::
        If possible, when creating the `OpticsParameters` to initiate this class, please do so
        by providing the ``TWISS`` values calculated at the center of elements. This is done by
        giving the flag `centre=true` to the ``TWISS`` command in ``MAD-X``, for instance. If
        this isn't done, a warning will be issued that one might observe some slight discrepancies
        against ``MAD-X`` result values.

    Attributes:
        beam_parameters (BeamParameters): the beam parameters to use for the calculations.
        optics (OpticsParameters): the optics parameters to use for the calculations.
        ibs_growth_rates (IBSGrowthRates): the computed IBS growth rates. This self-updates
            when they are computed with the `growth_rates` method.
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        super().__init__(beam_params, optics)

    def _Gamma(
        self,
        geom_epsx: float,
        geom_epsy: float,
        sigma_delta: float,
        bunch_length: float,
        bunched: bool = True,
    ) -> float:
        r"""
        .. versionadded:: 0.3.0

        Computes :math:`\Gamma`, the 6-dimensional invariant phase space volume of a bunched beam.

        Args:
            epsx (float): horizontal geometric emittance in [m].
            epxy (float): vertical geometric emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): the bunch length in [m].
            bunched (bool): whether the beam is bunched or not (coasting). Defaults to `True`.

        Returns:
            The computed :math:`\Gamma` value.
        """
        # fmt: off
        if bunched is True:
            return (
                (2 * np.pi)**3
                * (self.beam_parameters.beta_rel * self.beam_parameters.gamma_rel)**3
                * (self.beam_parameters.particle_mass_eV * 1e-3)**3  # use mass in MeV like in .growth_rates method (the m^3 terms cancel out)
                * geom_epsx
                * geom_epsy
                * sigma_delta
                * bunch_length
            )
        else:  # we have coasting beam
            return (
                4 * np.pi**(5/2)
                * (self.beam_parameters.beta_rel * self.beam_parameters.gamma_rel)**3
                * (self.beam_parameters.particle_mass_eV * 1e-3)**3  # use mass in MeV like in .growth_rates method (the m^3 terms cancel out)
                * geom_epsx
                * geom_epsy
                * sigma_delta
                * self.optics.circumference
            )
        # fmt: on

    def _a(self, geom_epsx: float, geom_epsy: float, sigma_delta: float) -> ArrayLike:
        """Computes the a term of Table 1 in the MAD-X note."""
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self.beam_parameters.beta_rel  # relativistic beta
        gamma: float = self.beam_parameters.gamma_rel  # relativistic gamma
        betx_over_epsx: np.ndarray = self.optics.betx / geom_epsx  # beta_x / eps_x term
        bety_over_epsy: np.ndarray = self.optics.bety / geom_epsy  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: np.ndarray = self.optics.dx * beta
        Dy: np.ndarray = self.optics.dy * beta
        Dpx: np.ndarray = self.optics.dpx * beta
        Dpy: np.ndarray = self.optics.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: np.ndarray = phi(self.optics.betx, self.optics.alfx, Dx, Dpx)
        phiy: np.ndarray = phi(self.optics.bety, self.optics.alfy, Dy, Dpy)
        Hx: np.ndarray = (Dx**2 + self.optics.betx**2 * phix**2) / self.optics.betx
        Hy: np.ndarray = (Dy**2 + self.optics.bety**2 * phiy**2) / self.optics.bety
        # ----------------------------------------------------------------------------------------------
        a: np.ndarray = (
            gamma**2 * (Hx / geom_epsx + Hy / geom_epsy)
            + gamma**2 / (sigma_delta**2)
            + (betx_over_epsx + bety_over_epsy)
        )
        return a

    def _b(self, geom_epsx: float, geom_epsy: float, sigma_delta: float) -> ArrayLike:
        """Computes the b term of Table 1 in the MAD-X note."""
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self.beam_parameters.beta_rel  # relativistic beta
        gamma: float = self.beam_parameters.gamma_rel  # relativistic gamma
        betxbety: np.ndarray = self.optics.betx * self.optics.bety  # beta_x * beta_y term
        epsxepsy: np.ndarray = geom_epsx * geom_epsy  # eps_x * eps_y term
        betx_over_epsx: np.ndarray = self.optics.betx / geom_epsx  # beta_x / eps_x term
        bety_over_epsy: np.ndarray = self.optics.bety / geom_epsy  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: np.ndarray = self.optics.dx * beta
        Dy: np.ndarray = self.optics.dy * beta
        Dpx: np.ndarray = self.optics.dpx * beta
        Dpy: np.ndarray = self.optics.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: np.ndarray = phi(self.optics.betx, self.optics.alfx, Dx, Dpx)
        phiy: np.ndarray = phi(self.optics.bety, self.optics.alfy, Dy, Dpy)
        # ----------------------------------------------------------------------------------------------
        b: np.ndarray = (
            (betx_over_epsx + bety_over_epsy)
            * (
                (gamma**2 * Dx**2) / (geom_epsx * self.optics.betx)
                + (gamma**2 * Dy**2) / (geom_epsy * self.optics.bety)
                + gamma**2 / sigma_delta**2
            )
            + betxbety * gamma**2 * (phix**2 + phiy**2) / (epsxepsy)
            + (betxbety / epsxepsy)
        )
        return b

    def _c(self, geom_epsx: float, geom_epsy: float, sigma_delta: float) -> ArrayLike:
        """Computes the c term of Table 1 in the MAD-X note."""
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self.beam_parameters.beta_rel  # relativistic beta
        gamma: float = self.beam_parameters.gamma_rel  # relativistic gamma
        betxbety: np.ndarray = self.optics.betx * self.optics.bety  # beta_x * beta_y term
        epsxepsy: np.ndarray = geom_epsx * geom_epsy  # eps_x * eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: np.ndarray = self.optics.dx * beta
        Dy: np.ndarray = self.optics.dy * beta
        # ----------------------------------------------------------------------------------------------
        c: np.ndarray = (betxbety / (epsxepsy)) * (
            (gamma**2 * Dx**2) / (geom_epsx * self.optics.betx)
            + (gamma**2 * Dy**2) / (geom_epsy * self.optics.bety)
            + gamma**2 / sigma_delta**2
        )
        return c

    def _ax(self, geom_epsx: float, geom_epsy: float, sigma_delta: float) -> ArrayLike:
        """Computes the ax term of Table 1 in the MAD-X note."""
        # ----------------------------------------------------------------------------------------------
        # We define new shorter names for a lot of arrays, for clarity of the expressions below
        betx: np.ndarray = self.optics.betx  # horizontal beta-functions
        bety: np.ndarray = self.optics.bety  # vertical beta-functions
        epsx: float = geom_epsx  # horizontal geometric emittance
        epsy: float = geom_epsy  # vertical geometric emittance
        sigd: float = sigma_delta  # momentum spread
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self.beam_parameters.beta_rel  # relativistic beta
        gamma: float = self.beam_parameters.gamma_rel  # relativistic gamma
        betx_over_epsx: np.ndarray = betx / epsx  # beta_x / eps_x term
        bety_over_epsy: np.ndarray = bety / epsy  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: np.ndarray = self.optics.dx * beta
        Dy: np.ndarray = self.optics.dy * beta
        Dpx: np.ndarray = self.optics.dpx * beta
        Dpy: np.ndarray = self.optics.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: np.ndarray = phi(self.optics.betx, self.optics.alfx, Dx, Dpx)
        phiy: np.ndarray = phi(self.optics.bety, self.optics.alfy, Dy, Dpy)
        Hx: np.ndarray = (Dx**2 + self.optics.betx**2 * phix**2) / self.optics.betx
        Hy: np.ndarray = (Dy**2 + self.optics.bety**2 * phiy**2) / self.optics.bety
        # ----------------------------------------------------------------------------------------------
        ax: np.ndarray = (  # double checked formula vs table - not same values as tx1 * cprime!
            2 * gamma**2 * (Hx / epsx + Hy / epsy + 1 / sigd**2)
            - (betx * Hy) / (Hx * epsy)
            + (betx / (Hx * gamma**2)) * (2 * betx_over_epsx - bety_over_epsy - gamma**2 / sigd**2)
            - 2 * betx_over_epsx
            - bety_over_epsy
            + (betx / (Hx * gamma**2)) * (6 * betx_over_epsx * gamma**2 * phix**2)
        )
        return ax

    def _bx(self, geom_epsx: float, geom_epsy: float, sigma_delta: float) -> ArrayLike:
        """Computes the bx term of Table 1 in the MAD-X note."""
        # ----------------------------------------------------------------------------------------------
        # We define new shorter names for a lot of arrays, for clarity of the expressions below
        betx: np.ndarray = self.optics.betx  # horizontal beta-functions
        bety: np.ndarray = self.optics.bety  # vertical beta-functions
        epsx: float = geom_epsx  # horizontal geometric emittance
        epsy: float = geom_epsy  # vertical geometric emittance
        sigd: float = sigma_delta  # momentum spread
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self.beam_parameters.beta_rel  # relativistic beta
        gamma: float = self.beam_parameters.gamma_rel  # relativistic gamma
        betx_over_epsx: np.ndarray = betx / epsx  # beta_x / eps_x term
        bety_over_epsy: np.ndarray = bety / epsy  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: np.ndarray = self.optics.dx * beta
        Dy: np.ndarray = self.optics.dy * beta
        Dpx: np.ndarray = self.optics.dpx * beta
        Dpy: np.ndarray = self.optics.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: np.ndarray = phi(self.optics.betx, self.optics.alfx, Dx, Dpx)
        phiy: np.ndarray = phi(self.optics.bety, self.optics.alfy, Dy, Dpy)
        Hx: np.ndarray = (Dx**2 + self.optics.betx**2 * phix**2) / self.optics.betx
        Hy: np.ndarray = (Dy**2 + self.optics.bety**2 * phiy**2) / self.optics.bety
        # ----------------------------------------------------------------------------------------------
        bx: np.ndarray = (  # double checked formula vs table - not same values as tx2 * cprime
            (betx_over_epsx + bety_over_epsy)
            * (gamma**2 * Hx / epsx + gamma**2 * Hy / epsy + gamma**2 / sigd**2)
            - gamma**2 * (betx_over_epsx**2 * phix**2 + bety_over_epsy**2 * phiy**2)
            + betx_over_epsx * (betx_over_epsx - 4 * bety_over_epsy)
            + (betx / (Hx * gamma**2))
            * (
                (gamma**2 / sigd**2) * (betx_over_epsx - 2 * bety_over_epsy)
                + betx_over_epsx * bety_over_epsy
                + 6 * betx_over_epsx * bety_over_epsy * gamma**2 * phix**2
                + gamma**2 * (2 * bety_over_epsy**2 * phiy**2 - betx_over_epsx**2 * phix**2)
            )
            + ((betx * Hy) / (epsy * Hx)) * (betx_over_epsx - 2 * bety_over_epsy)
        )
        return bx

    def _ay(self, geom_epsx: float, geom_epsy: float, sigma_delta: float) -> ArrayLike:
        """Computes the ay term of Table 1 in the MAD-X note."""
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self.beam_parameters.beta_rel  # relativistic beta
        gamma: float = self.beam_parameters.gamma_rel  # relativistic gamma
        betx_over_epsx: np.ndarray = self.optics.betx / geom_epsx  # beta_x / eps_x term
        bety_over_epsy: np.ndarray = self.optics.bety / geom_epsy  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: np.ndarray = self.optics.dx * beta
        Dy: np.ndarray = self.optics.dy * beta
        Dpx: np.ndarray = self.optics.dpx * beta
        Dpy: np.ndarray = self.optics.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: np.ndarray = phi(self.optics.betx, self.optics.alfx, Dx, Dpx)
        phiy: np.ndarray = phi(self.optics.bety, self.optics.alfy, Dy, Dpy)
        Hx: np.ndarray = (Dx**2 + self.optics.betx**2 * phix**2) / self.optics.betx
        Hy: np.ndarray = (Dy**2 + self.optics.bety**2 * phiy**2) / self.optics.bety
        # ----------------------------------------------------------------------------------------------
        ay: np.ndarray = (
            -(gamma**2)
            * (
                Hx / geom_epsx
                + 2 * Hy / geom_epsy
                + (self.optics.betx * Hy) / (self.optics.bety * geom_epsx)
                + 1 / sigma_delta**2
            )
            + 2 * gamma**4 * Hy / self.optics.bety * (Hy / geom_epsy + Hx / geom_epsx)
            + 2 * gamma**4 * Hy / (self.optics.bety * sigma_delta**2)
            - (betx_over_epsx - 2 * bety_over_epsy)
            + (6 * bety_over_epsy * gamma**2 * phiy**2)
        )
        return ay

    def _by(self, geom_epsx: float, geom_epsy: float, sigma_delta: float) -> ArrayLike:
        """Computes the by term of Table 1 in the MAD-X note."""
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self.beam_parameters.beta_rel  # relativistic beta
        gamma: float = self.beam_parameters.gamma_rel  # relativistic gamma
        betxbety: np.ndarray = self.optics.betx * self.optics.bety  # beta_x * beta_y term
        epsxepsy: np.ndarray = geom_epsx * geom_epsy  # eps_x * eps_y term
        betx_over_epsx: np.ndarray = self.optics.betx / geom_epsx  # beta_x / eps_x term
        bety_over_epsy: np.ndarray = self.optics.bety / geom_epsy  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: np.ndarray = self.optics.dx * beta
        Dy: np.ndarray = self.optics.dy * beta
        Dpx: np.ndarray = self.optics.dpx * beta
        Dpy: np.ndarray = self.optics.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: np.ndarray = phi(self.optics.betx, self.optics.alfx, Dx, Dpx)
        phiy: np.ndarray = phi(self.optics.bety, self.optics.alfy, Dy, Dpy)
        Hx: np.ndarray = (Dx**2 + self.optics.betx**2 * phix**2) / self.optics.betx
        Hy: np.ndarray = (Dy**2 + self.optics.bety**2 * phiy**2) / self.optics.bety
        # ----------------------------------------------------------------------------------------------
        by: np.ndarray = (
            gamma**2 * (bety_over_epsy - 2 * betx_over_epsx) * (Hx / geom_epsx + 1 / sigma_delta**2)
            + gamma**2 * Hy / geom_epsy * (bety_over_epsy - 4 * betx_over_epsx)
            + (betxbety / epsxepsy)
            + gamma**2 * (2 * betx_over_epsx**2 * phix**2 - bety_over_epsy**2 * phiy**2)
            + gamma**4
            * Hy
            / self.optics.bety
            * (betx_over_epsx + bety_over_epsy)
            * (Hy / geom_epsy + 1 / sigma_delta**2)
            + gamma**4 * Hx * Hy / (self.optics.bety * geom_epsx) * (betx_over_epsx + bety_over_epsy)
            - gamma**4
            * Hy
            / self.optics.bety
            * (betx_over_epsx**2 * phix**2 + bety_over_epsy**2 * phiy**2)
            + 6 * gamma**2 * phiy**2 * betx_over_epsx * bety_over_epsy
        )
        return by

    def _az(self, geom_epsx: float, geom_epsy: float, sigma_delta: float) -> ArrayLike:
        """Computes the al term of Table 1 in the MAD-X note."""
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self.beam_parameters.beta_rel  # relativistic beta
        gamma: float = self.beam_parameters.gamma_rel  # relativistic gamma
        betx_over_epsx: np.ndarray = self.optics.betx / geom_epsx  # beta_x / eps_x term
        bety_over_epsy: np.ndarray = self.optics.bety / geom_epsy  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: np.ndarray = self.optics.dx * beta
        Dy: np.ndarray = self.optics.dy * beta
        Dpx: np.ndarray = self.optics.dpx * beta
        Dpy: np.ndarray = self.optics.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: np.ndarray = phi(self.optics.betx, self.optics.alfx, Dx, Dpx)
        phiy: np.ndarray = phi(self.optics.bety, self.optics.alfy, Dy, Dpy)
        Hx: np.ndarray = (Dx**2 + self.optics.betx**2 * phix**2) / self.optics.betx
        Hy: np.ndarray = (Dy**2 + self.optics.bety**2 * phiy**2) / self.optics.bety
        # ----------------------------------------------------------------------------------------------
        az: np.ndarray = (
            2 * gamma**2 * (Hx / geom_epsx + Hy / geom_epsy + 1 / sigma_delta**2)
            - betx_over_epsx
            - bety_over_epsy
        )
        return az

    def _bz(self, geom_epsx: float, geom_epsy: float, sigma_delta: float) -> ArrayLike:
        """Computes the bl term of Table 1 in the MAD-X note."""
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self.beam_parameters.beta_rel  # relativistic beta
        gamma: float = self.beam_parameters.gamma_rel  # relativistic gamma
        betx_over_epsx: np.ndarray = self.optics.betx / geom_epsx  # beta_x / eps_x term
        bety_over_epsy: np.ndarray = self.optics.bety / geom_epsy  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: np.ndarray = self.optics.dx * beta
        Dy: np.ndarray = self.optics.dy * beta
        Dpx: np.ndarray = self.optics.dpx * beta
        Dpy: np.ndarray = self.optics.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: np.ndarray = phi(self.optics.betx, self.optics.alfx, Dx, Dpx)
        phiy: np.ndarray = phi(self.optics.bety, self.optics.alfy, Dy, Dpy)
        Hx: np.ndarray = (Dx**2 + self.optics.betx**2 * phix**2) / self.optics.betx
        Hy: np.ndarray = (Dy**2 + self.optics.bety**2 * phiy**2) / self.optics.bety
        # ----------------------------------------------------------------------------------------------
        bz: np.ndarray = (
            (betx_over_epsx + bety_over_epsy)
            * gamma**2
            * (Hx / geom_epsx + Hy / geom_epsy + 1 / sigma_delta**2)
            - 2 * betx_over_epsx * bety_over_epsy
            - gamma**2 * (betx_over_epsx**2 * phix**2 + bety_over_epsy**2 * phiy**2)
        )
        return bz

    def _constants(
        self,
        geom_epsx: float,
        geom_epsy: float,
        sigma_delta: float,
        bunch_length: float,
        bunched: bool = True,
    ) -> Tuple[float, ArrayLike, ArrayLike, float]:
        r"""
        .. versionadded:: 0.3.0

        Computes the constant terms of Eq (8) in :cite:`CERN:Antoniou:Revision_IBS_MADX`.
        Returned are four terms: first the constant common to all planes, then the horizontal,
        vertical and longitudinal terms (in brackets in Eq (8)).

        The common constant and the longitudinal constant are floats. The horizontal and vertical
        terms are arrays, with one value per element in the lattice (as they depend on :math:`H_x`
        and :math:`\beta_y`, respectively).

        Args:
            epsx (float): horizontal geometric emittance in [m].
            epxy (float): vertical geometric emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): the bunch length in [m].
            bunched (bool): whether the beam is bunched or not (coasting). Defaults to `True`.

        Returns:
            Four variables corresponding to the common, horizontal, vertical and longitudinal
            constants of Eq (8) in :cite:`CERN:Antoniou:Revision_IBS_MADX`.
        """
        # ----------------------------------------------------------------------------------------------
        # fmt: off
        # We define new shorter names for a lot of arrays, for clarity of the expressions below
        betx: np.ndarray = self.optics.betx  # horizontal beta-functions
        bety: np.ndarray = self.optics.bety  # vertical beta-functions
        alfx: np.ndarray = self.optics.alfx  # horizontal alpha-functions
        epsx: float = geom_epsx  # horizontal geometric emittance
        epsy: float = geom_epsy  # vertical geometric emittance
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self.beam_parameters.beta_rel  # relativistic beta
        gamma: float = self.beam_parameters.gamma_rel  # relativistic gamma
        bety_over_epsy: np.ndarray = bety / epsy  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: np.ndarray = self.optics.dx * beta
        Dpx: np.ndarray = self.optics.dpx * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: np.ndarray = phi(betx, alfx, Dx, Dpx)
        Hx: np.ndarray = (Dx**2 + betx**2 * phix**2) / betx
        # ----------------------------------------------------------------------------------------------
        # Compute the Coulomb logarithm and the common constant term in Eq (8) (the first fraction)
        coulomb_logarithm: float = self.coulomb_log(geom_epsx, geom_epsy, sigma_delta, bunch_length, bunched)
        common_constant_term: float = (
            np.pi**2
            * self.beam_parameters.particle_classical_radius_m**2
            * c
            * (self.beam_parameters.particle_mass_eV * 1e-3)** 3  # use mass in MeV like in ._Gamma method (the m^3 terms cancel out)
            * self.beam_parameters.n_part
            * coulomb_logarithm
            / (self.beam_parameters.gamma_rel * self._Gamma(geom_epsx, geom_epsy, sigma_delta, bunch_length, bunched))
        )
        # ----------------------------------------------------------------------------------------------
        # fmt: on
        # Compute the plane-dependent constants (in brackets) for each plane of Eq (8) in the MAD-X note
        const_x: np.ndarray = gamma**2 * Hx / epsx
        const_y: np.ndarray = bety_over_epsy
        const_z: float = gamma**2 / sigma_delta**2
        # ----------------------------------------------------------------------------------------------
        # Return the four terms now
        return common_constant_term, const_x, const_y, const_z

    def growth_rates(
        self,
        epsx: float,
        epsy: float,
        sigma_delta: float,
        bunch_length: float,
        bunched: bool = True,
        normalized_emittances: bool = False,
        integration_intervals: int = 17,
    ) -> IBSGrowthRates:
        r"""
        .. versionadded:: 0.3.0

        Computes the ``IBS`` growth rates, named :math:`T_x, T_y` and :math:`T_z` in this code
        base. These correspond to the :math:`1 / \tau` term, for each plane :math:`x, y` and
        :math:`z`, respectively. The instance attribute `self.ibs_growth_rates` is automatically
        updated with the results of this method when it is called.

        .. warning::
            When creating the `OpticsParameters` to initiate this class, please do so by providing
            the ``TWISS`` values calculated at the center of elements. This is done by giving the
            flag `centre=true` to the ``TWISS`` command in ``MAD-X`` for instance. If not, one might
            observe some slight discrepancies with the ``MAD-X`` values.

        .. tip::
            The calculation is done according to the following steps, which are related to different
            equations in :cite:`CERN:Antoniou:Revision_IBS_MADX`:

                - Adjusts the :math:`D_x, D_y, D^{\prime}_{x}, D^{\prime}_{y}` terms (multiply by :math:`\beta_{rel}`) to be in the :math:`pt` frame.
                - Computes the various terms from Table 1 of the MAD-X note.
                - Computes the Coulomb logarithm and the common constant term (first fraction) of Eq (8).
                - Defines the integrands of integrals in Eq (8) of the MAD-X note.
                - Defines sub-intervals and integrates the above over all of them, getting growth rates at each element in the lattice.
                - Averages the results over the full circumference of the machine.

        .. note::
            Both geometric or normalized emittances can be given as input to this function, and it is assumed
            the user provides geomettric emittances. If normalized ones are given the `normalized_emittances`
            parameter should be set to `True` (it defaults to `False`). Internally, a conversion is done to
            geometric emittances, which are used in the computations.

        Args:
            epsx (float): horizontal geometric or normalized emittance in [m].
            epsy (float): vertical geometric or normalized emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): the bunch length in [m].
            bunched (bool): whether the beam is bunched or not (coasting). Defaults to `True`.
            normalized_emittances (bool): whether the provided emittances are
                normalized or not. Defaults to `False` (assume geometric emittances).
            integration_intervals (int): the number of sub-intervals to use when integrating the
                integrands of Eq (8) of the MAD-X note. Please DO NOT change this parameter unless
                you know exactly what you are doing, as you might affect convergence. Defaults to 17.

        Returns:
            An `IBSGrowthRates` object with the computed growth rates for each plane.
        """
        # ----------------------------------------------------------------------------------------------
        # Make sure we are working with geometric emittances
        geom_epsx = epsx if normalized_emittances is False else self._geometric_emittance(epsx)
        geom_epsy = epsy if normalized_emittances is False else self._geometric_emittance(epsy)
        # ----------------------------------------------------------------------------------------------
        # We warn the user in case the TWISS was not centered - but keep going
        if self.optics._is_centered is False:
            LOGGER.warning("Twiss was not calculated at center of elements")
            warnings.warn(
                "The provided Twiss was calculated at the exit of the elements, but a centered version is "
                "desired. You might notice some discrepancies with the results from MAD-X itself."
            )
        # fmt: off
        # All of the following (when type annotated as np.ndarray), hold one value per element in the lattice
        # ----------------------------------------------------------------------------------------------
        # Getting the arrays from Table 1 of the MAD-X note
        LOGGER.debug("Computing terms from Table 1 of the MAD-X note")
        a: np.ndarray = self._a(geom_epsx, geom_epsy, sigma_delta)    # This is 'a' in MAD-X fortran code
        b: np.ndarray = self._b(geom_epsx, geom_epsy, sigma_delta)    # This is 'b' in MAD-X fortran code
        c: np.ndarray = self._c(geom_epsx, geom_epsy, sigma_delta)    # This is 'cprime' in MAD-X fortran code
        ax: np.ndarray = self._ax(geom_epsx, geom_epsy, sigma_delta)  # This is 'tx1 * cprime / bracket_x' in MAD-X fortran code
        bx: np.ndarray = self._bx(geom_epsx, geom_epsy, sigma_delta)  # This is 'tx2 * cprime / bracket_x' in MAD-X fortran code
        ay: np.ndarray = self._ay(geom_epsx, geom_epsy, sigma_delta)  # This is 'ty1 * cprime' in MAD-X fortran code
        by: np.ndarray = self._by(geom_epsx, geom_epsy, sigma_delta)  # This is 'ty2 * cprime' in MAD-X fortran code
        az: np.ndarray = self._az(geom_epsx, geom_epsy, sigma_delta)  # This is 'tl1 * cprime' in MAD-X fortran code
        bz: np.ndarray = self._bz(geom_epsx, geom_epsy, sigma_delta)  # This is 'tl2 * cprime' in MAD-X fortran code                                   
        # ----------------------------------------------------------------------------------------------
        # Getting the constant term and the bracket terms from Eq (8) of the MAD-X note
        LOGGER.debug("Computing common constant term and bracket terms from Eq (8) of the MAD-X note")
        common_constant_term, bracket_x, bracket_y, bracket_z = self._constants(
            geom_epsx, geom_epsy, sigma_delta, bunch_length, bunched
        )
        # ----------------------------------------------------------------------------------------------
        # Defining the integrands from Eq (8) of the MAD-X note, for each plane (remember these functions
        # are vectorised since a, b, c, ax, bx, ay, by are all arrays). The bracket terms are included.
        LOGGER.debug("Defining integrands of Eq (8) of the MAD-X note")
        def Ix_integrand_vec(_lambda: float) -> ArrayLike:
            numerator: np.ndarray = bracket_x * np.sqrt(_lambda) * (ax * _lambda + bx)
            denominator: np.ndarray = (_lambda**3 + a * _lambda**2 + b * _lambda + c) ** (3 / 2)
            return numerator / denominator

        def Iy_integrand_vec(_lambda: float) -> ArrayLike:
            numerator: np.ndarray = bracket_y * np.sqrt(_lambda) * (ay * _lambda + by)
            denominator: np.ndarray = (_lambda**3 + a * _lambda**2 + b * _lambda + c) ** (3 / 2)
            return numerator / denominator

        def Iz_integrand_vec(_lambda: float) -> ArrayLike:
            numerator: np.ndarray = bracket_z * np.sqrt(_lambda) * (az * _lambda + bz)
            denominator: np.ndarray = (_lambda**3 + a * _lambda**2 + b * _lambda + c) ** (3 / 2)
            return numerator / denominator
        # ----------------------------------------------------------------------------------------------
        # Defining a function to perform the integrating, which is done sub-interval by sub-interval
        def calculate_integral_vec(func: Callable) -> ArrayLike:
            """Defines limits of intervals, then goes over all intervals and performs the integration
            of the provided function on each one. At each step, we add the intermediate values to the
            final result, which is returned."""
            nb_elements: int = ax.size
            result: np.ndarray = np.zeros(nb_elements)

            # The following two hold the values for starts and ends of sub-intervals on which to integrate
            interval_starts = np.array([10**i for i in np.arange(0, int(integration_intervals) - 1)])
            interval_ends = np.array([10**i for i in np.arange(1, int(integration_intervals))])

            # Now we loop over the intervals and integrate the function on each one, using scipy
            # We add the intermediate integration result of each interval to our final result
            for start, end in zip(interval_starts, interval_ends):
                integrals, _ = quad_vec(func, start, end)  # integrals is an array
                result += integrals
            return result
        # ----------------------------------------------------------------------------------------------
        # fmt: on
        # Now we loop over the lattice and compute the integrals at each element
        LOGGER.debug("Computing integrals of Eq (8) of the MAD-X note - at each element in the lattice")
        Tx_array: np.ndarray = calculate_integral_vec(Ix_integrand_vec)
        Ty_array: np.ndarray = calculate_integral_vec(Iy_integrand_vec)
        Tz_array: np.ndarray = calculate_integral_vec(Iz_integrand_vec)
        # ----------------------------------------------------------------------------------------------
        # Don't forget to multiply by the common constant term here
        LOGGER.debug("Including common constant term of Eq (8) of the MAD-X note")
        Tx_array *= common_constant_term
        Ty_array *= common_constant_term
        Tz_array *= common_constant_term
        # ----------------------------------------------------------------------------------------------
        # Compute the final growth rates for each plane as an average | make sure to convert back to float
        # Interpolate the growth rates through the lattice for the average calculation below
        LOGGER.debug("Interpolating intermediate growth rates through the lattice")
        _tx = interp1d(self.optics.s, Tx_array)
        _ty = interp1d(self.optics.s, Ty_array)
        _tz = interp1d(self.optics.s, Tz_array)
        # ----------------------------------------------------------------------------------------------
        # To get a better average, we interpolate the array over the s coordinate, and then integrate this
        # interpolated function over the whole ring.
        LOGGER.debug("Getting average growth rates over the lattice")
        with warnings.catch_warnings():  # Catch and ignore the scipy.integrate.IntegrationWarning
            warnings.simplefilter("ignore", category=UserWarning)
            Tx: float = float(quad(_tx, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference)
            Ty: float = float(quad(_ty, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference)
            Tz: float = float(quad(_tz, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference)
        result = IBSGrowthRates(Tx, Ty, Tz)
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.ibs_growth_rates = result
        return result
