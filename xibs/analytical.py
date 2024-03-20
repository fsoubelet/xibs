r"""
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
        - It is assumed that no betatron coupling is present in the machine (or very little, in the order of :math:`\left| C^{-} \right| \le 10^{-4}`).

    Should these assumptions not be satisfied, the results provided by these calculations are not to be entirely discarded but might not be totally accurate.
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

from xibs.formulary import _percent_change, phi
from xibs.inputs import BeamParameters, OpticsParameters

LOGGER = getLogger(__name__)

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


@dataclass
class _SynchrotronRadiationInputs:
    """
    .. versionadded:: 0.6.0

    Container dataclass for SR input into emittance evolutions.

    Args:
        equilibrium_epsx (float): the horizontal equilibrium emittance from synchrotron radiation and quantum excitation in [m].
        equilibrium_epsy (float): the vertical equilibrium emittance from synchrotron radiation and quantum excitation in [m].
        equilibrium_sigma_delta (float): the equilibrium momentum spread from synchrotron radiation and quantum excitation.
        tau_x (float): the horizontal damping time from synchrotron radiation, in [s].
        tau_y (float): the vertical damping time from synchrotron radiation, in [s].
        tau_z (float): the longitudinal damping time from synchrotron radiation, in [s].
    """

    equilibrium_epsx: float
    equilibrium_epsy: float
    equilibrium_sigma_delta: float
    tau_x: float
    tau_y: float
    tau_z: float


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
        # Private attribute tracking the number of growth rates computations
        self._number_of_growth_rates_computations: int = 0

    def __str__(self) -> str:
        has_growth_rates = isinstance(
            self.ibs_growth_rates, IBSGrowthRates
        )  # False if default for value of None
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
        # Compute sigmas in each dimension (start from sigma_delta to get sige needed in the formula)
        sigma_x_cm = 100 * np.sqrt(
            geom_epsx * _bx_bar + (_dx_bar * sigma_delta * self.beam_parameters.beta_rel**2) ** 2
        )
        sigma_y_cm = 100 * np.sqrt(
            geom_epsy * _by_bar + (_dy_bar * sigma_delta * self.beam_parameters.beta_rel**2) ** 2
        )
        sigma_t_cm = 100 * bunch_length
        # ----------------------------------------------------------------------------------------------
        # Calculate beam volume to get density (in cm^{-3}) then Debye length
        if bunched is True:  # bunched beam
            volume = 8.0 * np.sqrt(np.pi**3) * sigma_x_cm * sigma_y_cm * sigma_t_cm
        else:  # coasting beam
            volume = 4.0 * np.pi * sigma_x_cm * sigma_y_cm * 100 * self.optics.circumference
        density = self.beam_parameters.n_part / volume
        debyul = (
            743.4 * np.sqrt(TempeV / density) / abs(self.beam_parameters.particle_charge)
        )  # abs for negative charges!
        # ----------------------------------------------------------------------------------------------
        # Calculate 'rmin' as larger of classical distance of closest approach or quantum mechanical
        # diffraction limit from nuclear radius
        rmincl = 1.44e-7 * self.beam_parameters.particle_charge**2 / TempeV
        rminqm = (
            hbar * c * 1e5 / (2.0 * np.sqrt(2e-3 * Etrans * self.beam_parameters.particle_mass_eV * 1e-9))
        )  # particle mass needed in GeV
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
        auto_recompute_rates_percent: float = None,
        **kwargs,
    ) -> Tuple[float, float, float, float]:
        r"""
        .. versionadded:: 0.2.0

        Analytically computes the new emittances after a given time step `dt` has
        ellapsed, from initial values, based on the ``IBS`` growth rates.

        .. warning::
            This calculation is done by building on the ``IBS`` growth rates. If the
            latter have not been computed yet, this method will raise an error. Please
            remember to call the instance's `growth_rates` method first.

        .. hint::
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


        .. admonition:: Synchrotron Radiation

            Synchrotron Radiation can play a significant role in the evolution of the emittances
            in certain scenarios, particularly for leptons. One can include the contribution of
            SR to this calculation by providing several keyword arguments corresponding to the
            equilibrium emittances and damping times from SR and quantum excitation. See the list
            of expected kwargs below. A :ref:`dedicated section in the FAQ <xibs-faq-sr-inputs>`
            provides information on how to obtain these values from ``Xsuite`` or ``MAD-X``.

            In case this contribution is included, then the calculation is modified from the one
            shown above, and goes according to :cite:`BOOK:Wolski:Beam_dynamics` (Eq (13.64)) or
            :cite:`CAS:Martini:IBS_Anatomy_Theory` (Eq (135)):

            .. math::

                T_{x,y,z} &= 1 / \tau_{x,y,z}^{\mathrm{IBS}}

                \varepsilon_{x,y}^{N+1} &= \left[ - \varepsilon_{x,y}^{\mathrm{SR}eq} + \left( \varepsilon_{x,y}^{\mathrm{SR}eq} + \frac{\varepsilon_{x,y}^{N}}{2 \tau_{x,y}^{\mathrm{IBS}}} \tau_{x,y}^{\mathrm{SR}} - 1 \right) * e^{2 t \left( \frac{1}{2 \tau_{x,y}^{\mathrm{IBS}}} - \frac{1}{\tau_{x,y}^{\mathrm{SR}}} \right)} \right] / \left( \frac{\tau_{x,y}^{\mathrm{SR}}}{2 \tau_{x,y}^{\mathrm{IBS}}} - 1 \right)

                {\sigma_{\delta, z}^{N+1}}^2 &= \left[ - {\sigma_{\delta, z}^{\mathrm{SR}eq}}^2 + \left( {\sigma_{\delta, z}^{\mathrm{SR}eq}}^2 + \frac{{\sigma_{\delta, z}^{N}}^2}{2 \tau_{z}^{\mathrm{IBS}}} \tau_{z}^{\mathrm{SR}} - 1 \right) * e^{2 t \left( \frac{1}{2 \tau_{z}^{\mathrm{IBS}}} - \frac{1}{\tau_{z}^{\mathrm{SR}}} \right)} \right] / \left( \frac{\tau_{z}^{\mathrm{SR}}}{2 \tau_{z}^{\mathrm{IBS}}} - 1 \right)


        Args:
            epsx (float): horizontal geometric or normalized emittance in [m].
            epsy (float): vertical geometric or normalized emittance in [m].
            sigma_delta (float): momentum spread.
            dt (float, optional): the time interval to use, in [s]. Default to the inverse
                of the revolution frequency, :math:`1 / f_{rev}`.
            bunch_length (float): the bunch length in [m].
            normalized_emittances (bool): whether the provided emittances are
                normalized or not. Defaults to `False` (assume geometric emittances).
            auto_recompute_rates_percent (float): Optional. If given, a check is performed to
                determine if an update of the growth rates is necessary, in which case it will
                be done computing the emittance evolutions. **Please provide a value as a percentage
                of the emittance change**. For instance, if one provides `12`, a check is made to see
                if any quantity would changed by more than 12%, and if so the growth rates are
                automatically recomputed to be as up-to-date as possible before returning the new values.
                Defaults to `None` (no checks done, no auto-recomputing).
            **kwargs: If keyword arguments are provided, they are considered inputs for the
                inclusion of synchrotron radiation in the calculation, and the following are
                expected, case-insensitively:

                    - `sr_equilibrium_epsx` (float)
                        the horizontal equilibrium emittance from synchrotron radiation
                        and quantum excitation, in [m]. Should be the same type (geometric
                        or normalized) as `epsx` and `epsy`.

                    - `sr_equilibrium_epsy` (float)
                        the vertical equilibrium emittance from synchrotron radiation and
                        quantum excitation, in [m]. Should be the same type (geometric or
                        normalized) as `epsx` and `epsy`.

                    - `sr_equilibrium_sigma_delta` (float)
                        the equilibrium momentum spread from synchrotron radiation and
                        quantum excitation.

                    - `sr_tau_x` (float)
                        the horizontal damping time from synchrotron radiation, in [s]
                        (should be the same unit as `dt`).

                    - `sr_tau_y` (float)
                        the vertical damping time from synchrotron radiation, in [s]
                        (should be the same unit as `dt`).

                    - `sr_tau_z` (float)
                        the longitudinal damping time from synchrotron radiation, in [s]
                        (should be the same unit as `dt`).


        Raises:
            AttributeError: if the ``IBS`` growth rates have not yet been computed.

        Returns:
            A tuple with the new horizontal & vertical geometric emittances, the new
            momentum spread and the new bunch length, after the time step has ellapsed.
        """
        # ----------------------------------------------------------------------------------------------
        # Check the kwargs and potentially get the arguments to include synchrotron radiation
        include_synchrotron_radiation = False
        if len(kwargs.keys()) >= 1:  # lets' not check with 'is not None' since default {} kwargs is not None
            LOGGER.debug("Kwargs present, assuming synchrotron radiation is to be included")
            include_synchrotron_radiation = True
            sr_inputs: _SynchrotronRadiationInputs = self._get_synchrotron_radiation_kwargs(**kwargs)
        # ----------------------------------------------------------------------------------------------
        # Make sure we are working with geometric emittances (also for SR inputs if given)
        geom_epsx = epsx if normalized_emittances is False else self._geometric_emittance(epsx)
        geom_epsy = epsy if normalized_emittances is False else self._geometric_emittance(epsy)
        if include_synchrotron_radiation is True:
            sr_eq_geom_epsx = (
                sr_inputs.equilibrium_epsx
                if normalized_emittances is False
                else self._geometric_emittance(sr_inputs.equilibrium_epsx)
            )
            sr_eq_geom_epsy = (
                sr_inputs.equilibrium_epsy
                if normalized_emittances is False
                else self._geometric_emittance(sr_inputs.equilibrium_epsy)
            )
            sr_eq_sigma_delta = sr_inputs.equilibrium_sigma_delta
        # ----------------------------------------------------------------------------------------------
        # Check that the IBS growth rates have been computed beforehand
        if self.ibs_growth_rates is None and auto_recompute_rates_percent is None:
            LOGGER.error("Attempted to compute emittance evolution without having computed growth rates.")
            raise AttributeError(
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
        if include_synchrotron_radiation is False:  # the basic calculation
            new_epsx, new_epsy, new_sigma_delta, new_bunch_length = self._evolution_without_sr(
                geom_epsx, geom_epsy, sigma_delta, bunch_length, dt
            )
        else:  # the modified calculation with Synchrotron Radiation contribution
            new_epsx, new_epsy, new_sigma_delta, new_bunch_length = self._evolution_with_sr(
                geom_epsx,
                geom_epsy,
                sigma_delta,
                bunch_length,
                dt,
                sr_eq_geom_epsx,
                sr_eq_geom_epsy,
                sr_eq_sigma_delta,
                sr_inputs.tau_x,
                sr_inputs.tau_y,
                sr_inputs.tau_z,
            )
        # ----------------------------------------------------------------------------------------------
        # Make sure we return the same type of emittances as the user provided
        new_epsx = new_epsx if normalized_emittances is False else self._normalized_emittance(new_epsx)
        new_epsy = new_epsy if normalized_emittances is False else self._normalized_emittance(new_epsy)
        # ----------------------------------------------------------------------------------------------
        # If we check gfor autoupdate and there is an change of more than self.auto_recompute_rates_percent %
        if isinstance(auto_recompute_rates_percent, (int, float)):
            if (
                abs(_percent_change(epsx, new_epsx)) > auto_recompute_rates_percent
                or abs(_percent_change(epsy, new_epsy)) > auto_recompute_rates_percent
                or abs(_percent_change(sigma_delta, new_sigma_delta)) > auto_recompute_rates_percent
                or abs(_percent_change(bunch_length, new_bunch_length)) > auto_recompute_rates_percent
            ):
                LOGGER.debug(
                    f"One value would change by more than {auto_recompute_rates_percent}%, "
                    "updating growth rates before re-computing evolutions."
                )
                bunched = kwargs.get("bunched", True)  # get the bunched value if provided
                self.growth_rates(epsx, epsy, sigma_delta, bunch_length, bunched, normalized_emittances)
                # And now we need to recompute the evolutions since the growth rates have been updated
                if include_synchrotron_radiation is False:  # the basic calculation
                    new_epsx, new_epsy, new_sigma_delta, new_bunch_length = self._evolution_without_sr(
                        geom_epsx, geom_epsy, sigma_delta, bunch_length, dt
                    )
                else:  # the modified calculation with Synchrotron Radiation contribution
                    new_epsx, new_epsy, new_sigma_delta, new_bunch_length = self._evolution_with_sr(
                        geom_epsx,
                        geom_epsy,
                        sigma_delta,
                        bunch_length,
                        dt,
                        sr_eq_geom_epsx,
                        sr_eq_geom_epsy,
                        sr_eq_sigma_delta,
                        sr_inputs.tau_x,
                        sr_inputs.tau_y,
                        sr_inputs.tau_z,
                    )
        # ----------------------------------------------------------------------------------------------
        return float(new_epsx), float(new_epsy), float(new_sigma_delta), float(new_bunch_length)

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

    def _get_synchrotron_radiation_kwargs(self, **kwargs) -> _SynchrotronRadiationInputs:
        r"""
        .. versionadded:: 0.6.0

        Called in `.emittance_evolution`. Gets the expected synchrotron radiation kwargs,
        and returns them as a dataclass. Will first convert to lowercase so the user does
        not have to worry about this.

        Raises:
            KeyError: if any of the expected kwargs is not provided.

        Returns:
            The parsed keyword arguments as a `_SynchrotronRadiationInputs` object.
        """
        lowercase_kwargs = {key.lower(): value for key, value in kwargs.items()}
        expected_keys = [
            "sr_equilibrium_epsx",
            "sr_equilibrium_epsy",
            "sr_equilibrium_sigma_delta",
            "sr_tau_x",
            "sr_tau_y",
            "sr_tau_z",
        ]
        if any(key not in lowercase_kwargs.keys() for key in expected_keys):
            LOGGER.error("Missing expected synchrotron radiation kwargs, see raised error message.")
            raise KeyError(
                "Not all expected synchrotron radiationkwargs were provided.\n"
                f"Expected: {expected_keys}, provided: {lowercase_kwargs.keys()}"
            )
        return _SynchrotronRadiationInputs(
            equilibrium_epsx=lowercase_kwargs["sr_equilibrium_epsx"],
            equilibrium_epsy=lowercase_kwargs["sr_equilibrium_epsy"],
            equilibrium_sigma_delta=lowercase_kwargs["sr_equilibrium_sigma_delta"],
            tau_x=lowercase_kwargs["sr_tau_x"],
            tau_y=lowercase_kwargs["sr_tau_y"],
            tau_z=lowercase_kwargs["sr_tau_z"],
        )

    def _evolution_without_sr(
        self, geom_epsx: float, geom_epsy: float, sigma_delta: float, bunch_length: float, dt: float
    ) -> Tuple[float, float, float, float]:
        """Just the calculation, called by main method when relevant."""
        new_epsx: float = geom_epsx * np.exp(dt * self.ibs_growth_rates.Tx)
        new_epsy: float = geom_epsy * np.exp(dt * self.ibs_growth_rates.Ty)
        new_sigma_delta: float = sigma_delta * np.exp(dt * 0.5 * self.ibs_growth_rates.Tz)
        new_bunch_length: float = bunch_length * np.exp(dt * 0.5 * self.ibs_growth_rates.Tz)
        return float(new_epsx), float(new_epsy), float(new_sigma_delta), float(new_bunch_length)

    def _evolution_with_sr(
        self,
        geom_epsx: float,
        geom_epsy: float,
        sigma_delta: float,
        bunch_length: float,
        dt: float,
        sr_eq_geom_epsx: float,
        sr_eq_geom_epsy: float,
        sr_eq_sigma_delta: float,
        sr_taux: float,
        sr_tauy: float,
        sr_tauz: float,
    ) -> Tuple[float, float, float, float]:
        """Just the calculation, called by main method when relevant."""
        # fmt: off
        new_epsx: float = (
            - sr_eq_geom_epsx
            + (sr_eq_geom_epsx + geom_epsx * (self.ibs_growth_rates.Tx / 2 * sr_taux - 1.0))
                * np.exp(2 * dt * (self.ibs_growth_rates.Tx / 2 - 1 / sr_taux))
        ) / (self.ibs_growth_rates.Tx / 2 * sr_taux - 1)
        new_epsy: float = (
            - sr_eq_geom_epsy
            + (sr_eq_geom_epsy + geom_epsy * (self.ibs_growth_rates.Ty / 2 * sr_tauy - 1))
                * np.exp(2 * dt * (self.ibs_growth_rates.Ty / 2 - 1 / sr_tauy))
        ) / (self.ibs_growth_rates.Ty / 2 * sr_tauy - 1)
        # For longitudinal properties, compute the square to avoid too messy code
        new_sigma_delta_square: float = (
            - (sr_eq_sigma_delta**2)
            + (sr_eq_sigma_delta**2 + sigma_delta**2 * (self.ibs_growth_rates.Tz / 2 * sr_tauz - 1))
                * np.exp(2 * dt * (self.ibs_growth_rates.Tz / 2 - 1 / sr_tauz))
        ) / (self.ibs_growth_rates.Tz / 2 * sr_tauz - 1)
        new_bunch_length_square: float = (
            - (sr_eq_sigma_delta**2)
            + (sr_eq_sigma_delta**2 + bunch_length**2 * (self.ibs_growth_rates.Tz / 2 * sr_tauz - 1))
                * np.exp(2 * dt * (self.ibs_growth_rates.Tz / 2 - 1 / sr_tauz))
        ) / (self.ibs_growth_rates.Tz / 2 * sr_tauz - 1)
        # And then simply get the square root of that for the final results
        new_sigma_delta: float = np.sqrt(new_sigma_delta_square)
        new_bunch_length: float = np.sqrt(new_bunch_length_square)
        # fmt: on
        return float(new_epsx), float(new_epsy), float(new_sigma_delta), float(new_bunch_length)


# ----- Classes to Compute Analytical IBS Growth Rates ----- #


class NagaitsevIBS(AnalyticalIBS):
    r"""
    .. versionadded:: 0.2.0

    A single class to compute Nagaitsev integrals (see
    :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`)
    and IBS growth rates. It initiates from a `BeamParameters` and an `OpticsParameters` objects.

    See the :ref:`Nagaitsev example <demo-analytical-nagaitsev>` for detailed usage, and the
    :ref:`Bjorken-Mtingwa example <demo-analytical-bjorken-mtingwa>` for a comparison to the
    Bjorken-Mtingwa formalism.

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

        .. hint::
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

        .. warning::
            Currently this calculation does not take into account vertical dispersion. Should you have
            any in your lattice, please use the BjorkenMtingwaIBS class instead, which supports it fully.
            Supporting vertical dispersion in NagaitsevIBS might be implemented in a future version.

        .. hint::
            The calculation is done according to the following steps, which are related to different
            equations in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`:

                - Get the Nagaitsev integrals from the instance attributes (integrals of Eq (30-32)).
                - Computes the Coulomb logarithm for the defined beam and optics parameters.
                - Compute the rest of the constant term of Eq (30-32).
                - Compute for each plane the full result of Eq (30-32), respectively.
                - Plug these into Eq (28) and divide by either :math:`\varepsilon_x, \varepsilon_y` or :math:`\sigma_{\delta}^{2}` (as relevant) to get :math:`1 / \tau`.

            **Note:** As one can see above, this calculation is done by building on the Nagaitsev integrals.
            If these have not been computed yet, this method will first log a message and compute them, then
            compute the growth rates.

        .. admonition:: Geometric or Normalized Emittances

            Both geometric or normalized emittances can be given as input to this function, and it is assumed
            the user provides geomettric emittances. If normalized ones are given the `normalized_emittances`
            parameter should be set to `True` (it defaults to `False`). Internally, a conversion is done to
            geometric emittances, which are used in the computations. For more information please see the
            following :ref:`section of the FAQ <xibs-faq-geom-norm-emittances>`.

        .. admonition:: Coasting Beams

            It is possible in this formalism to get an approximation in the case of coasting beams by providing
            `bunched=False`. This will as a bunch length :math:`C / 2 \pi` with C the circumference (or length)
            of the machine, and a warning will be logged for the user. Additionally the appropriate adjustement
            will be made in the Coulomb logarithm calculation, and the resulting growth rates will be divided by
            a factor 2 before being returned (see :cite:`ICHEA:Piwinski:IntraBeamScattering`). For fully accurate
            results in the case of coasting beams, please use the `BjorkenMtingwaIBS` class instead.

        Args:
            epsx (float): horizontal geometric or normalized emittance in [m].
            epsy (float): vertical geometric or normalized emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): the bunch length in [m].
            bunched (bool): whether the beam is bunched or not (coasting). Defaults to `True`. Please note
                that this will do an approximation using `bunch_length=C/(2*pi)`. For fully accurate results
                in the case of coasting beams, please use the BjorkenMtingwaIBS class instead.
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
            LOGGER.warning(
                "Using 'bunched=False' in this formalism makes the approximation of bunch length = C/(2*pi). "
                "Please use the BjorkenMtingwaIBS class for fully accurate results."
            )
            bunch_length: float = self.optics.circumference / (2 * np.pi)
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
        coulomb_logarithm = self.coulomb_log(geom_epsx, geom_epsy, sigma_delta, bunch_length, bunched)
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
        # Below: if coasting beams since we use bunch_length=C/(2*pi) we have to divide rates by 2 (see Piwinski)
        factor = 1.0 if bunched is True else 2.0
        Tx = float(Ix * full_constant_term / geom_epsx) / factor
        Ty = float(Iy * full_constant_term / geom_epsy) / factor
        Tz = float(Iz * full_constant_term / sigma_delta**2) / factor
        result = IBSGrowthRates(Tx, Ty, Tz)
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.ibs_growth_rates = result
        self._number_of_growth_rates_computations += 1
        return result


class BjorkenMtingwaIBS(AnalyticalIBS):
    r"""
    .. versionadded:: 0.3.0

    A single class to compute the IBS growth rates according to the `Bjorken & Mtingwa` formalism.
    The exact approach follows the ``MAD-X`` implementation, which has corrected B&M in order to
    take in consideration the vertical dispersion values (see the relevant note about the changes
    at :cite:`CERN:Antoniou:Revision_IBS_MADX`). It initiates from a `BeamParameters` and an
    `OpticsParameters` objects.

    See the :ref:`Bjorken-Mtingwa example <demo-analytical-bjorken-mtingwa>` for detailed usage,
    and the :ref:`Nagaitsev example <demo-analytical-nagaitsev>` for a comparison to the Nagaitsev
    formalism.

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

        .. hint::
            The calculation is done according to the following steps, which are related to different
            equations in :cite:`CERN:Antoniou:Revision_IBS_MADX`:

                - Adjusts the :math:`D_x, D_y, D^{\prime}_{x}, D^{\prime}_{y}` terms (multiply by :math:`\beta_{rel}`) to be in the :math:`pt` frame.
                - Computes the various terms from Table 1 of the MAD-X note.
                - Computes the Coulomb logarithm and the common constant term (first fraction) of Eq (8).
                - Defines the integrands of integrals in Eq (8) of the MAD-X note.
                - Defines sub-intervals and integrates the above over all of them, getting growth rates at each element in the lattice.
                - Averages the results over the full circumference of the machine.

        .. admonition:: Geometric or Normalized Emittances

            Both geometric or normalized emittances can be given as input to this function, and it is assumed
            the user provides geomettric emittances. If normalized ones are given the `normalized_emittances`
            parameter should be set to `True` (it defaults to `False`). Internally, a conversion is done to
            geometric emittances, which are used in the computations. For more information please see the
            following :ref:`section of the FAQ <xibs-faq-geom-norm-emittances>`.

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
        self._number_of_growth_rates_computations += 1
        return result
