"""
.. _xibs-kicks:

IBS: Applying Kicks
-------------------

Module with user-facing API to compute relevant terms to IBS kicks according to different formalism: simple and kinetic kicks.

In the simple formalism, the applied IBS kicks are determined from analytical IBS growth rates, which are computed internally (see :ref:`xibs-analytical`).
In the kinetic formalism, which adapts the kinetic theory of gases, the applied IBS kicks are determined from computed diffusion and friction terms.
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass
from logging import getLogger

import numpy as np

from numpy.typing import ArrayLike
from scipy.constants import c
from scipy.special import elliprd

from xibs.analytical import AnalyticalIBS, BjorkenMtingwaIBS, IBSGrowthRates, NagaitsevIBS
from xibs.formulary import (
    _bunch_length,
    _geom_epsx,
    _geom_epsy,
    _percent_change,
    _sigma_delta,
    _sigma_x,
    _sigma_y,
    phi,
)
from xibs.inputs import BeamParameters, OpticsParameters

LOGGER = getLogger(__name__)


# ----- Dataclasses to store results ----- #


@dataclass
class DiffusionCoefficients:
    """Container dataclass for kinetic IBS diffusion coefficients.

    Args:
        Dx (float): horizontal diffusion coefficient.
        Dy (float): vertical diffusion coefficient.
        Dz (float): longitudinal diffusion coefficient.
    """

    Dx: float
    Dy: float
    Dz: float


@dataclass
class FrictionCoefficients:
    """Container dataclass for kinetic IBS friction coefficients.

    Args:
        Fx (float): horizontal friction coefficient.
        Fy (float): vertical friction coefficient.
        Fz (float): longitudinal friction coefficient.
    """

    Fx: float
    Fy: float
    Fz: float


@dataclass
class IBSKickCoefficients:
    """
    Container dataclass for all IBS kick coefficients. These can be the coeffients from simple kicks,
    computed from analytical growth rates, or kinetic coefficients computed from the diffusion and
    friction ones according to :cite:`NuclInstr:Zenkevich:Kinetic_IBS`.

    Args:
        Kx (float): horizontal kick coefficient.
        Ky (float): vertical kick coefficient.
        Kz (float): longitudinal kick coefficient.
    """

    Kx: float
    Ky: float
    Kz: float


# ----- Abstract Base Class to Inherit from ----- #


class KickBasedIBS(ABC):
    r"""
    .. versionadded:: 0.5.0

    Abstract base class for kick-based IBS effects, from which all
    implementations inherit.

    Attributes:
        beam_parameters (BeamParameters): the beam parameters to use for IBS computations.
        optics (OpticsParameters): the optics parameters to use for the IBS computations.
        auto_recompute_coefficients_percent (float): Optional. If given, a check is performed after
            kicking the particles to determine if recomputing the kick coefficients is necessary, in
            which case it will be done before the next kick. **Please provide a value as a percentage
            of the emittance change**. For instance, if one provides `12` after kicking a check is
            done to see if the emittance changed by more than 12% in any plane, and if so the coefficients
            will be automatically recomputed before the next kick. Defaults to `None` (no checks done,
            no auto-recomputing).
        kick_coefficients (IBSKickCoefficients): the computed IBS kick coefficients. This
            attribute self-updates when they are computed with the `compute_kick_coefficients`
            method. It can also be set manually.
    """

    def __init__(
        self,
        beam_params: BeamParameters,
        optics: OpticsParameters,
        auto_recompute_coefficients_percent: float = None,
    ) -> None:
        self.beam_parameters: BeamParameters = beam_params
        self.optics: OpticsParameters = optics
        self.auto_recompute_coefficients_percent: float = auto_recompute_coefficients_percent
        # These coefficients self-update when computed, but can be overwritten by the user
        self.kick_coefficients: IBSKickCoefficients = None
        # Private flag to indicate if the coefficients need to be recomputed before the next kick
        self._need_to_recompute_coefficients: bool = False
        # Private attribute tracking the number of coefficients computations
        self._number_of_coefficients_computations: int = 0

    def __str__(self) -> str:
        has_kick_coefficients = isinstance(self.kick_coefficients, IBSKickCoefficients)
        auto_recomputes_coefficients = self.auto_recompute_coefficients_percent is not None
        return (
            f"{self.__class__.__name__} object for kick-based IBS calculations. "
            f"IBS kick coefficients computed: {has_kick_coefficients}. "
            f"Auto-recompute coefficients: {auto_recomputes_coefficients}."
        )

    def __repr__(self) -> str:
        return self.__str__()

    def line_density(self, particles: "xtrack.Particles", n_slices: int) -> ArrayLike:  # noqa: F821
        r"""
        .. versionadded:: 0.5.0

        Returns the "line density" of the `Particles` object, along its longitudinal axis, which
        corresponds to the :math:`\rho_t(t)` term in Eq (8) of :cite:`PRAB:Bruce:Simple_IBS_Kicks`.
        The density is used as a weight factor for the application of IBS kicks: particles in the
        denser parts of the bunch will receive a larger kick, and vice versa. See section III.C of
        the above reference details.

        .. hint::
            The calculation is done according to the following steps:

                - Gets the longitudinal coordinates of the active particles (state > 0) in the `Particles` object.
                - Determines coordinate cuts at front and back of the bunch, as well as slice width.
                - Determines bin edges and bin centers for the distribution for the chosen number of slices.
                - Computes a (normalized) histogram of the longitudinal coordinates, with the determined bins.
                - Computes and returns the line density :math:`\rho_t(t)`.

        Args:
            particles (xtrack.Particles): the `xtrack.Particles` object to compute the line density for.
            n_slices (int): the number of slices to use for the computation of the bins.

        Returns:
            An array with the density values for each slice / bin of the `Particles` object.
        """
        # ----------------------------------------------------------------------------------------------
        # Start with getting the nplike_lib from the particles' context, to compute on the context device
        nplike = particles._context.nplike_lib
        # ----------------------------------------------------------------------------------------------
        # Determine properties from longitudinal particles distribution: cuts, slice width, bunch length
        LOGGER.debug("Determining longitudinal particles distribution properties")
        zeta: ArrayLike = particles.zeta[particles.state > 0]  # careful to only consider active particles
        z_cut_head: float = nplike.max(zeta)  # z cut at front of bunch
        z_cut_tail: float = nplike.min(zeta)  # z cut at back of bunch
        slice_width: float = (z_cut_head - z_cut_tail) / n_slices  # slice width
        # ----------------------------------------------------------------------------------------------
        # Determine bin edges and bin centers for the distribution
        LOGGER.debug("Determining bin edges and bin centers for the distribution")
        bin_edges = nplike.linspace(
            z_cut_tail - 1e-7 * slice_width,
            z_cut_head + 1e-7 * slice_width,
            num=n_slices + 1,
            dtype=np.float64,
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        # ----------------------------------------------------------------------------------------------
        # Compute histogram on longitudinal distribution then compute and return line density
        counts_normed, bin_edges = nplike.histogram(zeta, bin_edges, density=True)  # density to normalize
        return nplike.interp(zeta, bin_centers, counts_normed)

    @abstractmethod
    def compute_kick_coefficients(
        self, particles: "xtrack.Particles", **kwargs  # noqa: F821
    ) -> IBSKickCoefficients:
        r"""
        .. versionadded:: 0.5.0

        Abstract method to determine the kick coefficients used in the determination of the IBS
        to be applied. It returns an `IBSKickCoefficients` object with the computed coefficients.

        Args:
            particles (xtrack.Particles): the particles to apply the IBS kicks to.
        """
        pass

    @abstractmethod
    def _apply_formalism_ibs_kick(
        self, particles: "xtrack.Particles", n_slices: int = 40  # noqa: F821
    ) -> None:
        r"""
        .. versionadded:: 0.5.0

        Abstract method to determine and apply IBS kicks to a `xtrack.Particles` object. The
        implementation details vary depending on the formalism implemented. This method is
        called in the `apply_ibs_kick` method.

        Args:
            particles (xtrack.Particles): the particles to apply the IBS kicks to.
        """
        pass

    def apply_ibs_kick(self, particles: "xtrack.Particles", n_slices: int = 40) -> None:  # noqa: F821
        r"""
        .. versionadded:: 0.5.0

        Compute and apply momenta kicks based on the provided `xtrack.Particles` object and the
        chosen ``IBS`` formalism. See the `_apply_formalism_ibs_kick` method for implementation details
        of the currently selected formalism.

        Args:
            particles (xtrack.Particles): the `xtrack.Particles` object to apply ``IBS`` kicks to.
            n_slices (int): the number of slices to use for the computation of the line density.
                Defaults to 40.

        Raises:
            AttributeError: if the ``IBS`` kick coefficients have not yet been computed.

        TODO: maybe instead of raising just compute the rates if they are not there. We could set
        the flag in self._check_coefficients_presence to True and it will be computed later on.
        """
        # ----------------------------------------------------------------------------------------------
        # Check that the kick coefficients have been computed beforehand
        self._check_coefficients_presence()
        # ----------------------------------------------------------------------------------------------
        # Check the auto-recompute flag and recompute coefficients if necessary
        if self._need_to_recompute_coefficients is True:
            LOGGER.info("Recomputing IBS kick coefficients before applying kicks")
            self.compute_kick_coefficients(particles)
            self._need_to_recompute_coefficients = False
        # ----------------------------------------------------------------------------------------------
        # Get and store pre-kick emittances if self.auto_recompute_coefficients_percent is set
        if isinstance(self.auto_recompute_coefficients_percent, (int, float)):
            _previous_bunch_length = _bunch_length(particles)
            _previous_sigma_delta = _sigma_delta(particles)
            # below we give index 0 as start / end of machine is kick location
            _previous_geom_epsx = _geom_epsx(particles, self.optics.betx[0], self.optics.dx[0])
            _previous_geom_epsy = _geom_epsy(particles, self.optics.bety[0], self.optics.dy[0])
        # ----------------------------------------------------------------------------------------------
        # Apply the kicks to the particles - the function implementation here is formalism-specific
        self._apply_formalism_ibs_kick(particles, n_slices)
        # ----------------------------------------------------------------------------------------------
        # Get post-kick emittances, check growth and set recompute flag if necessary (only if self.auto_recompute_coefficients_percent is set)
        # fmt: off
        if isinstance(self.auto_recompute_coefficients_percent, (int, float)):
            _new_bunch_length = _bunch_length(particles)
            _new_sigma_delta = _sigma_delta(particles)
            _new_geom_epsx = _geom_epsx(particles, self.optics.betx[0], self.optics.dx[0])
            _new_geom_epsy = _geom_epsy(particles, self.optics.bety[0], self.optics.dy[0])
            # If there is an increase / decrease of more than self.auto_recompute_coefficients_percent % in any plane, set the flag
            if (
                abs(_percent_change(_previous_bunch_length, _new_bunch_length)) > self.auto_recompute_coefficients_percent
                or abs(_percent_change(_previous_sigma_delta, _new_sigma_delta)) > self.auto_recompute_coefficients_percent
                or abs(_percent_change(_previous_geom_epsx, _new_geom_epsx)) > self.auto_recompute_coefficients_percent
                or abs(_percent_change(_previous_geom_epsy, _new_geom_epsy)) > self.auto_recompute_coefficients_percent
            ):
                LOGGER.debug(
                    f"One plane's emittance changed by more than {self.auto_recompute_coefficients_percent}%, "
                    "setting flag to recompute coefficients before next kick."
                )
                self._need_to_recompute_coefficients = True
        # fmt: on


# ----- Classes to Compute and Apply IBS Kicks ----- #


class SimpleKickIBS(KickBasedIBS):
    r"""
    .. versionadded:: 0.5.0

    A single class to compute the simple IBS kicks based on the analytical growth rates.
    The kicks are implemented according to :cite:`PRAB:Bruce:Simple_IBS_Kicks`, and
    provide a random distribution of momenta changes based on the growth rates, weighted
    by the line density of the bunch. The class initiates from a `BeamParameters` and an
    `OpticsParameters` objects.

    See the :ref:`simple kicks example <demo-simple-kicks>` for detailed usage.

    .. warning::
        Beware: this implementation is only valid **above** transition energy. Because
        this formalism implements a weighted random-component kick, it will *always* lead
        to emittance growth. Below transition it is common to observe negative growth rates,
        which would lead to emittance *shrinkage* and therefore the provided kick would have
        the wrong effect. It is also possible to obtain negative growth rates above transition
        in some scenarios, and internally this implementation sets the growth rate to 0 if it
        is found negative. When this happens, a message is logged to inform the user. For any
        machine operating below transition energy, the kinetic formalism should be used instead
        (see the `KineticKickIBS` class).

    .. hint::
        When determining kick coefficients (see the `compute_kick_coefficients` method),
        the analytical growth rates are computed. This is done using one of the analytical
        classes, which is determined internally based on the optics parameters (namely, the
        presence of vertical dispersion), and set as the `self.analytical_ibs` attribute.
        Choices are logged to the user. It is always possible to override this choice by
        manually setting the `self.analytical_ibs` attribute to an instance of the desired
        analytical implementation (to be found in `xibs.analytical`). It is also possible
        for the user to provide their own, custom-made analytical implementation, as long as
        it inherits from the `AnalyticalIBS` class and implements the API defined therein.

    Attributes:
        beam_parameters (BeamParameters): the beam parameters to use for IBS computations.
        optics (OpticsParameters): the optics parameters to use for the IBS computations.
        analytical_ibs (AnalyticalIBS): an internal analytical class for growth rates
            calculation, which is determined automatically. Can be overridden by the user
            by setting this attribute manually.
        auto_recompute_coefficients_percent (float): Optional. If given, a check is performed after
            kicking the particles to determine if recomputing the kick coefficients is necessary, in
            which case it will be done before the next kick. **Please provide a value as a percentage
            of the emittance increase**. For instance, if one provides `12` after kicking a check is
            done to see if the emittance grew by more than 12% in any plane, and if so the coefficients
            will be automatically recomputed before the next kick. Defaults to `None` (no checks done,
            no auto-recomputing).
        kick_coefficients (IBSKickCoefficients): the computed IBS kick coefficients. This
            self-updates when they are computed with the `compute_kick_coefficients` method.
            It can also be set manually.
    """

    def __init__(
        self,
        beam_params: BeamParameters,
        optics: OpticsParameters,
        auto_recompute_coefficients_percent: float = None,
    ) -> None:
        # fmt: off
        # First, we check that we are above transition and raise and error if not (not applicable)
        if optics.slip_factor <= 0:  # we are below transition (xsuite convention: slip factor > 0 above)
            LOGGER.error(
                "The provided optics parameters indicate that the machine is below transition, "
                "which is incompatible with SimpleKickIBS (see documentation). "
                "Use the kinetic formalism with KineticKickIBS instead."
            )
            raise NotImplementedError(
                "SimpleKickIBS is not compatible with machines operating below transition. "
                "Please see the documentation and use the kinetic formalism with KineticKickIBS instead."
            )
        # If we made it here, SimpleKickIBS is a valid implementation, let's instantiate from KickBasedIBS
        super().__init__(beam_params, optics, auto_recompute_coefficients_percent)  # also sets self.kick_coefficients (to None)
        # Analytical implementation for growth rates calculation, can be overridden by the user
        if np.count_nonzero(self.optics.dy) != 0:
            LOGGER.info("Non-zero vertical dispersion detected in the lattice, using Bjorken & Mtingwa formalism")
            self._analytical_ibs: AnalyticalIBS = BjorkenMtingwaIBS(beam_params, optics)
        else:
            LOGGER.info("No vertical dispersion in the lattice, using Nagaitsev formalism")
            self._analytical_ibs: AnalyticalIBS = NagaitsevIBS(beam_params, optics)
        LOGGER.info("This can be overridden manually, by explicitely setting the self.analytical_ibs attribute")
        # fmt: on

    @property
    def analytical_ibs(self) -> AnalyticalIBS:
        """The analytical IBS implementation used for growth rates calculation."""
        return self._analytical_ibs

    @analytical_ibs.setter
    def analytical_ibs(self, value: AnalyticalIBS) -> None:
        """The analytical_ibs has a setter so that .beam_params and .optics are updated when it is set."""
        # fmt: off
        LOGGER.debug("Overwriting the analytical ibs implementation used for growth rates calculation")
        self._analytical_ibs = value
        LOGGER.debug("Re-pointing the instance's beam and optics parameters to that of the new analytical implementation")
        self.beam_parameters = self.analytical_ibs.beam_parameters
        self.optics = self.analytical_ibs.optics
        # fmt: on

    def _check_coefficients_presence(self) -> None:
        """
        Call this before trying to apply kicks to first check the necessarykick coefficients are present.

        Raises:
            AttributeError: if the necessary ``IBS`` kick coefficients have not yet been computed.
        """
        if self.kick_coefficients is None:
            LOGGER.error("Attempted to apply IBS kick without having computed kick coefficients first.")
            raise AttributeError(
                "IBS kick coefficients have not been computed yet, cannot apply kick to particles.\n"
                "Please call the `compute_kick_coefficients` method first."
            )

    def compute_kick_coefficients(
        self, particles: "xtrack.Particles", **kwargs  # noqa: F821
    ) -> IBSKickCoefficients:
        r"""
        .. versionadded:: 0.5.0

        Computes the ``IBS`` kick coefficients, named :math:`K_x, K_y` and :math:`K_z` in this
        code base, from analytical growth rates. The coefficients correspond to the right-hand
        side of Eq (8) in :cite:`PRAB:Bruce:Simple_IBS_Kicks` without the line density :math:`\rho_t(t)`
        and random component :math:`r`.

        The kick coefficient corresponds to the scaling of the generated random distribution :math:`r` and
        is expressed as :math:`K_u = \sigma_{p_u} \sqrt{2 T^{-1}_{IBS_u} T_{rev} \sigma_t \sqrt{\pi}}`.

        .. note::
            This functionality is separate from the kick application as it internally
            triggers the computation of the analytical growth rates. Since this step
            is computationally intensive and one might not necessarily want to recompute
            the rates before every kick application.

        .. hint::
            The calculation is done according to the following steps, which are related to
            different terms in Eq (8) of :cite:`PRAB:Bruce:Simple_IBS_Kicks`:

                - Computes various properties from the non-lost particles in the bunch (:math:`\sigma_{x,y,\delta,t}`).
                - Computes the standard deviation of momenta for each plane (:math:`\sigma_{p_u}`).
                - Computes the constant term :math:`\sqrt{2 T_{rev} \sqrt{\pi}}`.
                - Computes the analytical growth rates :math:`T_{x,y,z}` (:math:`T^{-1}_{IBS_u}` in Eq (8)).
                - Computes, stores and returns the kick coefficients.

        Args:
            particles (xtrack.Particles): the particles to apply the IBS kicks to.
            **kwargs: any keyword arguments will be passed to the growth rates calculation call
                (`self.analytical_ibs.growth_rates`). Note that `epsx`, `epsy`, `sigma_delta`,
                and `bunch_length` are already provided as positional-only arguments.

        Returns:
            An `IBSKickCoefficients` object with the computed coefficients used for the kick application.
        """
        # ----------------------------------------------------------------------------------------------
        # Start with getting the nplike_lib from the particles' context, to compute on the context device
        nplike = particles._context.nplike_lib
        # ----------------------------------------------------------------------------------------------
        # Compute the momentum spread, bunch length and (geometric) emittances from the Particles object
        # Indexing at 0 as this end / start of machine is when we kick (after a line.track)
        LOGGER.debug("Computing emittances, momentum spread and bunch length from particles")
        bunch_length: float = _bunch_length(particles)
        sigma_delta: float = _sigma_delta(particles)
        geom_epsx: float = _geom_epsx(particles, self.optics.betx[0], self.optics.dx[0])
        geom_epsy: float = _geom_epsy(particles, self.optics.bety[0], self.optics.dy[0])
        # fmt: off
        # ----------------------------------------------------------------------------------------------
        # Computing standard deviation of (normalized) momenta, corresponding to sigma_{pu} in Eq (8) of reference
        # Normalized: for momentum we have to multiply with gamma = beta / (1 + alpha^2), beta is included in the
        # std of p[xy]. If bunch is rotated, the std takes from the "other plane" so we normalize to compensate.
        sigma_px_normalized: float = nplike.std(particles.px[particles.state > 0]) / nplike.sqrt(1 + self.optics.alfx[0]**2)
        sigma_py_normalized: float = nplike.std(particles.py[particles.state > 0]) / nplike.sqrt(1 + self.optics.alfy[0]**2)
        # ----------------------------------------------------------------------------------------------
        # Determine the "scaling factor", corresponding to 2 * sigma_t * sqrt(pi) in Eq (8) of reference
        zeta: np.ndarray = particles.zeta[particles.state > 0]  # careful to only consider active particles
        bunch_length_rms: float = nplike.std(zeta)  # rms bunch length in [m]
        scaling_factor: float = float(2 * nplike.sqrt(np.pi) * bunch_length_rms)
        # ----------------------------------------------------------------------------------------------
        # Computing the analytical IBS growth rates
        growth_rates: IBSGrowthRates = self.analytical_ibs.growth_rates(
            float(geom_epsx), float(geom_epsy), float(sigma_delta), float(bunch_length), **kwargs
        )
        Tx, Ty, Tz = astuple(growth_rates)  # on CPU
        # ----------------------------------------------------------------------------------------------
        # Making sure we do not have negative growth rates (see class docstring warning for detail)
        Tx = 0.0 if Tx < 0 else float(Tx)
        Ty = 0.0 if Ty < 0 else float(Ty)
        Tz = 0.0 if Tz < 0 else float(Tz)
        if any(rate == 0 for rate in (Tx, Ty, Tz)):
            LOGGER.info("At least one IBS growth rate was negative, and was set to 0")
        # ----------------------------------------------------------------------------------------------
        # Compute the kick coefficients - see function docstring for exact definition
        # For the longitudinal plane, since the values are computed from ΔP/P but applied to the ΔE/E
        # (the particles.delta in Xsuite), we need to multiply by beta_rel**2 to adapt
        LOGGER.debug("Computing and applying the kicks to the particles")
        Kx: float = sigma_px_normalized * nplike.sqrt(2 * scaling_factor * Tx / self.optics.revolution_frequency)
        Ky: float = sigma_py_normalized * nplike.sqrt(2 * scaling_factor * Ty / self.optics.revolution_frequency)
        Kz: float = sigma_delta * nplike.sqrt(2 * scaling_factor * Tz / self.optics.revolution_frequency) * self.beam_parameters.beta_rel**2
        result = IBSKickCoefficients(Kx, Ky, Kz)
        # fmt: on
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.kick_coefficients = result
        self._number_of_coefficients_computations += 1
        return result

    def _apply_formalism_ibs_kick(
        self, particles: "xtrack.Particles", n_slices: int = 40  # noqa: F821
    ) -> None:
        r"""
        .. versionadded:: 0.5.0

        Compute the momentum kick to apply based on the provided `xtrack.Particles` object and the
        analytical growth rates for the lattice. The kicks are implemented according to Eq (8) of
        :cite:`PRAB:Bruce:Simple_IBS_Kicks`.

        Args:
            particles (xtrack.Particles): the `xtrack.Particles` object to apply ``IBS`` kicks to.
            n_slices (int): the number of slices to use for the computation of the line density.
                Defaults to 40.
        """
        # ----------------------------------------------------------------------------------------------
        # Start with getting the the particles' context, to be able to move data on the context device
        context = particles._context
        # ----------------------------------------------------------------------------------------------
        # Compute the line density - this is the rho_t(t) term in Eq (8) of reference
        rho_t: ArrayLike = self.line_density(particles, n_slices)  # does NOT include _factor
        # ----------------------------------------------------------------------------------------------
        # Determining size of arrays for kicks to apply: only the non-lost particles in the bunch
        _size: int = particles.px[particles.state > 0].shape[0]  # same for py and delta
        # ----------------------------------------------------------------------------------------------
        # Determining kicks - this corresponds to the full result of Eq (8) of reference
        # In theory, .normal(0, 1, _size) * factor and .normal(0, factor, _size) are the same (try it) but
        # in practice the resulting kick is not physically accurate. The same is observed in C++ code (see
        # for instance BLonD) so we use the coefficients as scale here, which is correct and benchmarked.
        # fmt: off
        LOGGER.debug("Determining kicks to apply")
        RNG = np.random.default_rng()
        rho_t_ = context.nparray_from_context_array(rho_t)  # on CPU
        delta_px: np.ndarray = RNG.normal(loc=0, scale=float(self.kick_coefficients.Kx), size=_size) * np.sqrt(rho_t_)
        delta_py: np.ndarray = RNG.normal(loc=0, scale=float(self.kick_coefficients.Ky), size=_size) *  np.sqrt(rho_t_)
        delta_delta: np.ndarray = RNG.normal(loc=0, scale=float(self.kick_coefficients.Kz), size=_size) * np.sqrt(rho_t_)
        # ----------------------------------------------------------------------------------------------
        # Apply the kicks to the particles - just move the computed deltas to device and apply
        LOGGER.debug("Applying momenta kicks to the particles (on px, py and delta properties)")
        particles.px[particles.state > 0] += context.nparray_to_context_array(delta_px)
        particles.py[particles.state > 0] += context.nparray_to_context_array(delta_py)
        particles.delta[particles.state > 0] += context.nparray_to_context_array(delta_delta)
        # fmt: on


class KineticKickIBS(KickBasedIBS):
    r"""
    .. versionadded:: 0.7.0

    A single class to compute the IBS diffusion and friction coefficients according
    to the kinetic IBS formalism of :cite:`NuclInstr:Zenkevich:Kinetic_IBS`.
    The class initiates from a `BeamParameters` and an `OpticsParameters` objects.

    See the :ref:`kinetic kicks example <demo-kinetic-kicks>` for detailed usage.

    Attributes:
        beam_parameters (BeamParameters): the beam parameters to use for IBS computations.
        optics (OpticsParameters): the optics parameters to use for the IBS computations.
        diffusion_coefficients (DiffusionCoefficients): the computed diffusion coefficients
            from the kinetic theory. This attribute self-updates when coefficients are computed
            with the `compute_kick_coefficients` method. It can also be set manually.
        friction_coefficients (FrictionCoefficients): the computed friction coefficients
            from the kinetic theory. This attribute self-updates when coefficients are computed
            with the `compute_kick_coefficients` method. It can also be set manually.
        auto_recompute_coefficients_percent (float): Optional. If given, a check is performed after
            kicking the particles to determine if recomputing the kick coefficients is necessary, in
            which case it will be done before the next kick. **Please provide a value as a percentage
            of the emittance increase**. For instance, if one provides `12` after kicking a check is
            done to see if the emittance grew by more than 12% in any plane, and if so the coefficients
            will be automatically recomputed before the next kick. Defaults to `None` (no checks done,
            no auto-recomputing).
        kick_coefficients (IBSKickCoefficients): the computed IBS kick coefficients from
            the kinetic theory, determined from the diffusion and friction coefficients. This
            attribute self-updates when coefficients are computed with the `compute_kick_coefficients`
            method. It can also be set manually.
    """

    def __init__(
        self,
        beam_params: BeamParameters,
        optics: OpticsParameters,
        auto_recompute_coefficients_percent: float = None,
    ) -> None:
        super().__init__(
            beam_params, optics, auto_recompute_coefficients_percent
        )  # also sets self.kick_coefficients
        # These self-update when computed, but can be overwritten by the user
        self.diffusion_coefficients: DiffusionCoefficients = None
        self.friction_coefficients: FrictionCoefficients = None

    def compute_kick_coefficients(
        self, particles: "xtrack.Particles", **kwargs  # noqa: F821
    ) -> IBSKickCoefficients:
        r"""
        .. versionadded:: 0.7.0

        Computes the ``IBS`` kick coefficients, named :math:`K_x, K_y` and :math:`K_z` in this
        code base, from the friction and diffusion terms of the kinetic theory as expressed in
        :cite:`NuclInstr:Zenkevich:Kinetic_IBS`, and using terms from Nagaitsev's formalism
        (:cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`) as determined by M. Zampetakis
        (:cite:`CERN:Zampetakis:Implementation_IBS_Kicks`). This will compute both diffusion and
        friction coefficients from this formalism, which will be stored and updated internally
        into the `diffusion_coefficients` and `friction_coefficients` attributes. It returns
        an `IBSKickCoefficients` object with the computed coefficients (diffusion - friction).

        .. note::
            This functionality is separate from the kick application because it internally triggers
            the computation of the analytical growth rates, and we don't necessarily want to
            recompute these at every turn. Meanwhile, the kicks **should** be applied at every turn.

        .. hint::
            The calculation is done according to the following steps:

                - Computes various terms from :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation` as well as elliptic integrals.
                - Computes the :math:`D_{xx}, D_{xz}, D_{yy}, D_{zz}, K_x, K_y` and :math:`K_z` terms.
                - Computes diffusion and friction coefficients from the above, following :cite:`CERN:Zampetakis:Implementation_IBS_Kicks`.
                - Computes and returns kick coefficients (as the difference between diffusion and friction).

        Args:
            particles (xtrack.Particles): the particles to apply the IBS kicks to.
            **kwargs: if `bunched` is found in keyword arguments it will be passed to the
                coulomb logarithm calculation. A default value of `True` is used.

        Returns:
            An `IBSKickCoefficients` object with the computed coefficients used for the kick application.
        """
        # ----------------------------------------------------------------------------------------------
        # Start with getting the nplike_lib from the particles' context, to compute on the context device
        context = particles._context
        nplike = context.nplike_lib
        # ----------------------------------------------------------------------------------------------
        # Compute the momentum spread, bunch length and (geometric) emittances from the Particles object
        # Indexing at 0 as this end / start of machine is when we kick (after a line.track)
        LOGGER.debug("Computing emittances, momentum spread and bunch length from particles")
        bunch_length: float = _bunch_length(particles)
        sigma_delta: float = _sigma_delta(particles)
        geom_epsx: float = _geom_epsx(particles, self.optics.betx[0], self.optics.dx[0])
        geom_epsy: float = _geom_epsy(particles, self.optics.bety[0], self.optics.dy[0])
        sigma_x: float = _sigma_x(particles)
        sigma_y: float = _sigma_y(particles)
        # ----------------------------------------------------------------------------------------------
        # Moving some necessary arrays to device for later computation
        s: ArrayLike = context.nparray_to_context_array(self.optics.s)  # on device
        alfx: ArrayLike = context.nparray_to_context_array(self.optics.alfx)  # on device
        betx: ArrayLike = context.nparray_to_context_array(self.optics.betx)  # on device
        bety: ArrayLike = context.nparray_to_context_array(self.optics.bety)  # on device
        dx: ArrayLike = context.nparray_to_context_array(self.optics.dx)  # on device
        dpx: ArrayLike = context.nparray_to_context_array(self.optics.dpx)  # on device
        phix: ArrayLike = phi(betx, alfx, dx, dpx)  # on device as all dependent terms are
        # ----------------------------------------------------------------------------------------------
        # Allocating some values to simple variables for readability later
        gammar: float = self.beam_parameters.gamma_rel
        betar: float = self.beam_parameters.beta_rel
        n_part: int = self.beam_parameters.n_part
        classical_radius: float = self.beam_parameters.particle_classical_radius_m
        pi: float = np.pi
        circumference: float = self.optics.circumference
        # fmt: off
        # ----------------------------------------------------------------------------------------------
        # Computing the constants from Eq (18-21) in Nagaitsev paper - on device as all dependent terms are
        ax: ArrayLike = betx / geom_epsx
        ay: ArrayLike = bety / geom_epsy
        a_s: ArrayLike = ax * (dx**2 / betx**2 + phix**2) + 1 / sigma_delta**2
        a1: ArrayLike = (ax + gammar**2 * a_s) / 2.0
        a2: ArrayLike = (ax - gammar**2 * a_s) / 2.0
        sqrt_term = nplike.sqrt(a2**2 + gammar**2 * ax**2 * phix**2)
        # ----------------------------------------------------------------------------------------------
        # These are from Eq (22-24) in Nagaitsev paper, eigen values of A matrix (L matrix in B&M)
        # Also all on device as their dependent terms are on device
        lambda_1: ArrayLike = ay
        lambda_2: ArrayLike = a1 + sqrt_term
        lambda_3: ArrayLike = a1 - sqrt_term
        # ----------------------------------------------------------------------------------------------
        # These are the R_D terms to compute, from Eq (25-27) in Nagaitsev paper (at each element of the lattice)
        # Since cupy does not have an elliprd equivalent, we go back to CPU and let scipy handle this
        LOGGER.debug("Computing elliptic integrals R1, R2 and R3")
        lbd1_: ArrayLike = context.nparray_from_context_array(lambda_1)  # on CPU
        lbd2_: ArrayLike = context.nparray_from_context_array(lambda_2)  # on CPU
        lbd3_: ArrayLike = context.nparray_from_context_array(lambda_3)  # on CPU
        R1_: ArrayLike = elliprd(1 / lbd2_, 1 / lbd3_, 1 / lbd1_) / context.nparray_from_context_array(lambda_1)  # on CPU
        R2_: ArrayLike = elliprd(1 / lbd3_, 1 / lbd1_, 1 / lbd2_) / context.nparray_from_context_array(lambda_2)  # on CPU
        R3_: ArrayLike = 3 * np.sqrt(lbd1_ * lbd2_ / lbd3_) - lbd1_ * R1_ / lbd3_ - lbd2_ * R2_ / lbd3_           # on CPU
        # We transport these results back to device
        R1: ArrayLike = context.nparray_to_context_array(R1_)  # on device
        R2: ArrayLike = context.nparray_to_context_array(R2_)  # on device
        R3: ArrayLike = context.nparray_to_context_array(R3_)  # on device
        # ----------------------------------------------------------------------------------------------
        # Compute the coulomb logarithm from an analytical class then the rest of the constant term in
        # Eq (30-32) of Nagaitsev's paper - all this below are CPU computations
        analytical = NagaitsevIBS(self.beam_parameters, self.optics)  # the formalism does not matter
        bunched = kwargs.get("bunched", True)
        coulomb_logarithm: float = analytical.coulomb_log(geom_epsx, geom_epsy, sigma_delta, bunch_length, bunched)
        rest_of_constant_term: float = n_part * classical_radius**2 * c / (12 * pi * betar**3 * gammar**5 * bunch_length)
        full_constant_term: float = rest_of_constant_term * coulomb_logarithm
        # ----------------------------------------------------------------------------------------------
        # Computing the Dxx, Dxz, etc terms from Nagaitsev terms above, according to the expressions derived
        # by Michalis (see backup slides in his presentation at https://indico.cern.ch/event/1140639)
        # All below are on device as all dependent terms are on device
        Dzz: ArrayLike = 0.5 * gammar**2 * (2 * R1 + R2 * (1 + a2 / sqrt_term) + R3 * (1 - a2 / sqrt_term))  # on device
        Kz: ArrayLike = 1.0 * gammar**2 * (R2 * (1 - a2 / sqrt_term) + R3 * (1 + a2 / sqrt_term))            # on device
        Dxx: ArrayLike = 0.5 * (2 * R1 + R2 * (1 - a2 / sqrt_term) + R3 * (1 + a2 / sqrt_term))              # on device
        Kx: ArrayLike = 1.0 * (R2 * (1 + a2 / sqrt_term) + R3 * (1 - a2 / sqrt_term))                        # on device
        Dxz: ArrayLike = 3.0 * gammar**2 * phix**2 * ax * (R3 - R2) / sqrt_term                              # on device
        # ----------------------------------------------------------------------------------------------
        # Computing integrands for the diffusion and friction terms from the above (also from Michalis,
        # see slide 18 of his presentation for instance) - all of these are on device as all dependent terms are
        Dx_integrand: ArrayLike = betx / (circumference * sigma_x * sigma_y) * (Dxx + Dzz * (dx**2 / betx**2 + phix**2) + Dxz)  # on device
        Fx_integrand: ArrayLike = betx / (circumference * sigma_x * sigma_y) * (Kx + Kz * (dx**2 / betx**2 + phix**2))          # on device
        Dy_integrand: ArrayLike = bety / (circumference * sigma_x * sigma_y) * (R2 + R3)                                        # on device
        Fy_integrand: ArrayLike = bety / (circumference * sigma_x * sigma_y) * (2 * R1)                                         # on device
        Dz_integrand: ArrayLike = Dzz / (circumference * sigma_x * sigma_y)                                                     # on device
        Fz_integrand: ArrayLike = Kz / (circumference * sigma_x * sigma_y)                                                      # on device
        # ----------------------------------------------------------------------------------------------
        # Integrating them to obtain the diffusion and friction coefficients
        Dx: float = nplike.sum(Dx_integrand[:-1] * nplike.diff(s)) * full_constant_term / geom_epsx
        Dy: float = nplike.sum(Dy_integrand[:-1] * nplike.diff(s)) * full_constant_term / geom_epsy
        Dz: float = nplike.sum(Dz_integrand[:-1] * nplike.diff(s)) * full_constant_term / sigma_delta**2
        Fx: float = nplike.sum(Fx_integrand[:-1] * nplike.diff(s)) * full_constant_term / geom_epsx
        Fy: float = nplike.sum(Fy_integrand[:-1] * nplike.diff(s)) * full_constant_term / geom_epsy
        Fz: float = nplike.sum(Fz_integrand[:-1] * nplike.diff(s)) * full_constant_term / sigma_delta**2
        # ----------------------------------------------------------------------------------------------
        # Generate and store the coefficients in a DiffusionCoefficients and FrictionCoefficients object
        self.diffusion_coefficients = DiffusionCoefficients(Dx, Dy, Dz)
        self.friction_coefficients = FrictionCoefficients(Fx, Fy, Fz)
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results: kick coefficients
        result = IBSKickCoefficients(Dx - Fx, Dy - Fy, Dz - Fz)
        self.kick_coefficients = result
        self._number_of_coefficients_computations += 1
        return result

    def _check_coefficients_presence(self) -> None:
        """
        Call this before trying to apply kicks to first check the necessarykick coefficients are present.

        Raises:
            AttributeError: if the necessary ``IBS`` kick coefficients have not yet been computed.
        """
        if any(
            coeffs is None
            for coeffs in [self.kick_coefficients, self.diffusion_coefficients, self.friction_coefficients]
        ):
            LOGGER.error("Attempted to apply IBS kick without having computed kick coefficients first.")
            raise AttributeError(
                "IBS kick coefficients have not been computed yet, cannot apply kick to particles.\n"
                "Please call the `compute_kick_coefficients` method first."
            )

    def _apply_formalism_ibs_kick(
        self, particles: "xtrack.Particles", n_slices: int = 40  # noqa: F821
    ) -> None:
        r"""
        .. versionadded:: 0.7.0

        Computes the momentum kicks to apply based on the provided `xtrack.Particles` object and
        the previously computed kick coefficients. The kick is applied as described in
        :cite:`NuclInstr:Zenkevich:Kinetic_IBS` and :cite:`CERN:Zampetakis:Implementation_IBS_Kicks`.

        Args:
            particles (xtrack.Particles): the `xtrack.Particles` object to apply ``IBS`` kicks to.
            n_slices (int): the number of slices to use for the computation of the line density.
                Defaults to 40.
        """
        # ----------------------------------------------------------------------------------------------
        # Start with getting the nplike_lib from the particles' context, to compute on the context device
        context = particles._context
        nplike = context.nplike_lib
        # ----------------------------------------------------------------------------------------------
        # Compute the line density - this is the rho_t(t) term in Eq (8) of
        dt: float = 1 / self.optics.revolution_frequency
        rho_t: ArrayLike = self.line_density(particles, n_slices)  # computed on device
        # ----------------------------------------------------------------------------------------------
        # Compute the bunch_length * 2 * sqrt(pi) factor for the kicks
        bunch_length: float = _bunch_length(particles)
        factor: float = bunch_length * 2 * nplike.sqrt(np.pi)
        # ----------------------------------------------------------------------------------------------
        # Compute the momentum spread and standard deviation of (normalized) momenta from particles object
        # Normalized: for momentum we have to multiply with gamma = beta / (1 + alpha^2), beta is included in the
        # std of p[xy]. If bunch is rotated, the std takes from the "other plane" so we normalize to compensate.
        # fmt: off
        LOGGER.debug("Computing momentum spread and momenta's standard deviations")
        sigma_delta: float = _sigma_delta(particles)  # on device
        sigma_px_normalized: float = nplike.std(particles.px[particles.state > 0]) / nplike.sqrt(1 + self.optics.alfx[0]**2)  # on device
        sigma_py_normalized: float = nplike.std(particles.py[particles.state > 0]) / nplike.sqrt(1 + self.optics.alfy[0]**2)  # on device
        # ----------------------------------------------------------------------------------------------
        # Determining kicks from the friction forces (see referenced Michalis presentation)
        LOGGER.debug("Determining friction kicks")
        dev_px: ArrayLike = particles.px[particles.state > 0] - nplike.mean(particles.px[particles.state > 0])           # on device
        dev_py: ArrayLike = particles.py[particles.state > 0] - nplike.mean(particles.py[particles.state > 0])           # on device
        dev_delta: ArrayLike = particles.delta[particles.state > 0] - nplike.mean(particles.delta[particles.state > 0])  # on device
        Fx, Fy, Fz = astuple(self.friction_coefficients)
        delta_px_friction: ArrayLike = Fx * dev_px * dt * rho_t * factor        # on device
        delta_py_friction: ArrayLike = Fy * dev_py * dt * rho_t * factor        # on device
        delta_delta_friction: ArrayLike = Fz * dev_delta * dt * rho_t * factor  # on device
        # ----------------------------------------------------------------------------------------------
        # Determining kicks from the friction forces (see referenced Michalis presentation)
        # Since cupy does not provide a default_rng().normal method, we go back to CPU and let scipy handle this
        LOGGER.debug("Determining diffusion kicks")
        RNG = np.random.default_rng()
        _size: int = particles.px[particles.state > 0].shape[0]  # same for py and delta
        Dx, Dy, Dz = astuple(self.diffusion_coefficients)
        Dx, Dy, Dz = float(Dx), float(Dy), float(Dz)                            # on CPU
        sig_px_norm_ = context.nparray_from_context_array(sigma_px_normalized)  # on CPU
        sig_py_norm_ = context.nparray_from_context_array(sigma_py_normalized)  # on CPU
        sig_delta_ = context.nparray_from_context_array(sigma_delta)            # on CPU
        rho_t_ = context.nparray_from_context_array(rho_t)                      # on CPU
        factor_ = float(factor)                                                 # on CPU
        delta_px_diffusion: ArrayLike = sig_px_norm_ * np.sqrt(2 * dt * Dx) * RNG.normal(0, 1, _size) * np.sqrt(rho_t_ * factor_)   # on CPU
        delta_py_diffusion: ArrayLike = sig_py_norm_ * np.sqrt(2 * dt * Dy) * RNG.normal(0, 1, _size) * np.sqrt(rho_t_ * factor_)   # on CPU
        delta_delta_diffusion: ArrayLike = sig_delta_ * np.sqrt(2 * dt * Dz) * RNG.normal(0, 1, _size) * np.sqrt(rho_t_ * factor_)  # on CPU
        # ----------------------------------------------------------------------------------------------
        # Now we can apply all momenta kicks (friction and diffusion) to the particles - on device directly
        # fmt: on
        LOGGER.debug("Applying friction kicks to the particles (on px, py and delta properties)")
        particles.px[particles.state > 0] -= delta_px_friction
        particles.py[particles.state > 0] -= delta_py_friction
        particles.delta[particles.state > 0] -= delta_delta_friction
        LOGGER.debug("Applying diffusion kicks to the particles (on px, py and delta properties)")
        # Since the generated were done by numpy we make sure to convert to device first
        particles.px[particles.state > 0] += context.nparray_to_context_array(delta_px_diffusion)
        particles.py[particles.state > 0] += context.nparray_to_context_array(delta_py_diffusion)
        particles.delta[particles.state > 0] += context.nparray_to_context_array(delta_delta_diffusion)
