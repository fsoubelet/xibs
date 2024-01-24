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

from xibs.analytical import AnalyticalIBS, BjorkenMtingwaIBS, IBSGrowthRates, NagaitsevIBS
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
        kick_coefficients (IBSKickCoefficients): the computed IBS kick coefficients. This
            attribute self-updates when they are computed with the `compute_kick_coefficients`
            method. It can also be set manually.
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        self.beam_parameters: BeamParameters = beam_params
        self.optics: OpticsParameters = optics
        # These self-update when computed, but can be overwritten by the user
        self.kick_coefficients: IBSKickCoefficients = None

    def __str__(self) -> str:
        has_kick_coefficients = isinstance(self.kick_coefficients, IBSKickCoefficients)
        return (
            f"{self.__class__.__name__} object for kick-based IBS calculations."
            f"IBS kick coefficients computed: {has_kick_coefficients}"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def line_density(self, particles: "xpart.Particles", n_slices: int) -> ArrayLike:  # noqa: F821
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
            particles (xpart.Particles): the `xpart.Particles` object to compute the line density for.
            n_slices (int): the number of slices to use for the computation of the bins.

        Returns:
            An array with the density values for each slice / bin of the `Particles` object.
        """
        # ----------------------------------------------------------------------------------------------
        # Determine properties from longitudinal particles distribution: cuts, slice width, bunch length
        LOGGER.debug("Determining longitudinal particles distribution properties")
        zeta: np.ndarray = particles.zeta[particles.state > 0]  # careful to only consider active particles
        z_cut_head: float = np.max(zeta)  # z cut at front of bunch
        z_cut_tail: float = np.min(zeta)  # z cut at back of bunch
        slice_width: float = (z_cut_head - z_cut_tail) / n_slices  # slice width
        # ----------------------------------------------------------------------------------------------
        # Determine bin edges and bin centers for the distribution
        LOGGER.debug("Determining bin edges and bin centers for the distribution")
        bin_edges = np.linspace(
            z_cut_tail - 1e-7 * slice_width,
            z_cut_head + 1e-7 * slice_width,
            num=n_slices + 1,
            dtype=np.float64,
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        # ----------------------------------------------------------------------------------------------
        # Compute histogram on longitudinal distribution then compute and return line density
        counts_normed, bin_edges = np.histogram(zeta, bin_edges, density=True)  # density=True to normalize
        return np.interp(zeta, bin_centers, counts_normed)

    @abstractmethod
    def compute_kick_coefficients(
        self, particles: "xpart.Particles", **kwargs  # noqa: F821
    ) -> IBSKickCoefficients:
        r"""
        .. versionadded:: 0.5.0

        Abstract method to determine the kick coefficients used in the determination of the IBS
        to be applied. It returns an `IBSKickCoefficients` object with the computed coefficients.

        Args:
            particles (xpart.Particles): the particles to apply the IBS kicks to.
        """
        ...

    @abstractmethod
    def apply_ibs_kick(self, particles: "xpart.Particles") -> None:  # noqa: F821
        r"""
        .. versionadded:: 0.5.0

        Abstract method to determine and apply IBS kicks to a `xpart.Particles` object. The
        momenta kicks are determined from the `self.kick_coefficients` attribute.

        Args:
            particles (xpart.Particles): the particles to apply the IBS kicks to.

        Raises:
            AttributeError: if the ``IBS`` kick coefficients have not yet been computed.
        """
        ...


# ----- Classes to Compute and Apply IBS Kicks ----- #


class SimpleKickIBS(KickBasedIBS):
    r"""
    .. versionadded:: 0.5.0

    A single class to compute the simple IBS kicks based on the analytical growth rates.
    The kicks are implemented according to :cite:`PRAB:Bruce:Simple_IBS_Kicks`, and
    provide a random distribution of momenta changes based on the growth rates, weighted
    by the line density of the bunch. The class initiates from a `BeamParameters` and an
    `OpticsParameters` objects.

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
        kick_coefficients (IBSKickCoefficients): the computed IBS kick coefficients. This
            self-updates when they are computed with the `compute_kick_coefficients` method.
            It can also be set manually.
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        # fmt: off
        # First, we check that we are above transition and raise and error if not (not applicable)
        if optics.slip_factor <= 0:  # we are below transition (xsuite convention: slip factor > 0 above)
            LOGGER.error(
                "The provided optics parameters indication that the machine is below transition, "
                "which is incompatible with SimpleKickIBS (see documentation). "
                "Use the kinetic formalism with KineticKickIBS instead."
            )
            raise NotImplementedError(
                "SimpleKickIBS is not compatible with machine operating below transition. "
                "Please see the documentation and use the kinetic formalism with KineticKickIBS instead."
            )
        # If we made it here, SimpleKickIBS is a valid implementation, let's instantiate from KickBasedIBS
        super().__init__(beam_params, optics)  # also sets self.kick_coefficients (to None)
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

    def compute_kick_coefficients(
        self, particles: "xpart.Particles", **kwargs  # noqa: F821
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
            particles (xpart.Particles): the particles to apply the IBS kicks to.
            **kwargs: any keyword arguments will be passed to the growth rates calculation call
                (`self.analytical_ibs.growth_rates`). Note that `epsx`, `epsy`, `sigma_delta`,
                and `bunch_length` are already provided as positional-only arguments.

        Returns:
            An `IBSKickCoefficients` object with the computed coefficients used for the kick application.
        """
        # ----------------------------------------------------------------------------------------------
        # Compute the momentum spread, bunch length and (geometric) emittances from the Particles object
        LOGGER.debug("Computing emittances, momentum spread and bunch length from particles")
        bunch_length: float = float(np.std(particles.zeta[particles.state > 0]))
        sigma_delta: float = float(np.std(particles.delta[particles.state > 0]))
        sigma_x: float = float(np.std(particles.x[particles.state > 0]))
        sigma_y: float = float(np.std(particles.y[particles.state > 0]))
        # Indexing: we want the value where the bunch is and assume at start / end of machine which
        # is where / when we will apply the kick (do a line.track and then kick). To be modified when
        # when we include these things into xtrack, if we want an element that creates the kick.
        geom_epsx: float = (sigma_x**2 - (self.optics.dx[0] * sigma_delta) ** 2) / self.optics.betx[0]
        geom_epsy: float = (sigma_y**2 - (self.optics.dy[0] * sigma_delta) ** 2) / self.optics.bety[0]
        # fmt: off
        # ----------------------------------------------------------------------------------------------
        # Computing standard deviation of (normalized) momenta, corresponding to sigma_{pu} in Eq (8) of reference
        # Normalized: for momentum we have to multiply with gamma = beta / (1 + alpha^2), beta is included in the
        # std of p[xy]. If bunch is rotated, the std takes from the "other plane" so we normalize to compensate.
        sigma_px_normalized: float = np.std(particles.px[particles.state > 0]) / np.sqrt(1 + self.optics.alfx[0]**2)
        sigma_py_normalized: float = np.std(particles.py[particles.state > 0]) / np.sqrt(1 + self.optics.alfy[0]**2)
        # ----------------------------------------------------------------------------------------------
        # Determine the "scaling factor", corresponding to 2 * sigma_t * sqrt(pi) in Eq (8) of reference
        zeta: np.ndarray = particles.zeta[particles.state > 0]  # careful to only consider active particles
        bunch_length_rms: float = np.std(zeta)  # rms bunch length in [m]
        scaling_factor: float = float(2 * np.sqrt(np.pi) * bunch_length_rms)
        # ----------------------------------------------------------------------------------------------
        # Computing the analytical IBS growth rates
        growth_rates: IBSGrowthRates = self.analytical_ibs.growth_rates(
            geom_epsx, geom_epsy, sigma_delta, bunch_length, **kwargs
        )
        Tx, Ty, Tz = astuple(growth_rates)
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
        Kx: float = sigma_px_normalized * np.sqrt(2 * scaling_factor * Tx / self.optics.revolution_frequency)
        Ky: float = sigma_py_normalized * np.sqrt(2 * scaling_factor * Ty / self.optics.revolution_frequency)
        Kz: float = sigma_delta * np.sqrt(2 * scaling_factor * Tz / self.optics.revolution_frequency) * self.beam_parameters.beta_rel**2
        result = IBSKickCoefficients(Kx, Ky, Kz)
        # fmt: on
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.kick_coefficients = result
        return result

    def apply_ibs_kick(self, particles: "xpart.Particles", n_slices: int = 40) -> None:  # noqa: F821
        r"""
        .. versionadded:: 0.5.0

        Compute the momentum kick to apply based on the provided `xpart.Particles` object and the
        analytical growth rates for the lattice. The kicks are implemented according to Eq (8) of
        :cite:`PRAB:Bruce:Simple_IBS_Kicks`.

        Args:
            particles (xpart.Particles): the `xpart.Particles` object to apply ``IBS`` kicks to.
            n_slices (int): the number of slices to use for the computation of the line density.
                Defaults to 40.

        Raises:
            AttributeError: if the ``IBS`` kick coefficients have not yet been computed.
        """
        # ----------------------------------------------------------------------------------------------
        # Check that the kick coefficients have been computed beforehand
        if self.kick_coefficients is None:
            LOGGER.error("Attempted to apply IBS kick without having computed kick coefficients first.")
            raise AttributeError(
                "IBS kick coefficients have not been computed yet, cannot apply kick to particles.\n"
                "Please call the `compute_kick_coefficients` method first."
            )
        # fmt: off
        # ----------------------------------------------------------------------------------------------
        # Compute the line density - this is the rho_t(t) term in Eq (8) of reference
        rho_t: np.ndarray = self.line_density(particles, n_slices)  # does NOT include _factor
        # ----------------------------------------------------------------------------------------------
        # Determining size of arrays for kicks to apply: only the non-lost particles in the bunch
        _size: int = particles.px[particles.state > 0].shape[0]  # same for py and delta
        # ----------------------------------------------------------------------------------------------
        # Determining kicks - this corresponds to the full result of Eq (8) of reference
        # In theory, .normal(0, 1, _size) * factor and .normal(0, factor, _size) are the same (try it) but
        # in practice the resulting kick is not physically accurate. The same is observed in C++ code (see
        # for instance BLonD) so we use the coefficients as scale here, which is correct and benchmarked.
        LOGGER.debug("Determining kicks to apply")
        RNG = np.random.default_rng()
        delta_px: np.ndarray = RNG.normal(loc=0, scale=self.kick_coefficients.Kx, size=_size) * np.sqrt(rho_t)
        delta_py: np.ndarray = RNG.normal(loc=0, scale=self.kick_coefficients.Ky, size=_size) *  np.sqrt(rho_t)
        delta_delta: np.ndarray = RNG.normal(loc=0, scale=self.kick_coefficients.Kz, size=_size) * np.sqrt(rho_t)
        # fmt: on
        # ----------------------------------------------------------------------------------------------
        # Apply the kicks to the particles
        LOGGER.debug("Applying momenta kicks to the particles (on px, py and delta properties)")
        particles.px[particles.state > 0] += delta_px
        particles.py[particles.state > 0] += delta_py
        particles.delta[particles.state > 0] += delta_delta


# ---------------------------------------------------------------- #
# ----- THIS IS TESTING FOR NOW AND SHOULD / WILL BE REMOVED ----- #
# ---------------------------------------------------------------- #


# It does seem that Michalis for kinetic uses some of the R1, R2 etc terms from the Nagaitsev
# formalism for some reason? Will need to clarify with him.
class KineticKickIBS(KickBasedIBS):
    r"""
    .. versionadded:: 0.5.0

    A single class to compute the IBS diffusion and friction coefficients according
    to the kinetic IBS formalism of :cite:`NuclInstr:Zenkevich:Kinetic_IBS`.
    The class initiates from a `BeamParameters` and an `OpticsParameters` objects.

    TODO: reference, details etc.

    Attributes:
        beam_parameters (BeamParameters): the beam parameters to use for IBS computations.
        optics (OpticsParameters): the optics parameters to use for the IBS computations.
        diffusion_coefficients (DiffusionCoefficients): the computed diffusion coefficients
            from the kinetic theory. This attribute self-updates when coefficients are computed
            with the `compute_kick_coefficients` method. It can also be set manually.
        friction_coefficients (FrictionCoefficients): the computed friction coefficients
            from the kinetic theory. This attribute self-updates when coefficients are computed
            with the `compute_kick_coefficients` method. It can also be set manually.
        kick_coefficients (IBSKickCoefficients): the computed IBS kick coefficients from
            the kinetic theory, determined from the diffusion and friction coefficients. This
            attribute self-updates when coefficients are computed with the `compute_kick_coefficients`
            method. It can also be set manually.
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        super().__init__(beam_params, optics)  # also sets self.kick_coefficients
        # These self-update when computed, but can be overwritten by the user
        self.diffusion_coefficients: DiffusionCoefficients = None
        self.friction_coefficients: FrictionCoefficients = None

    def compute_kick_coefficients(
        self, particles: "xpart.Particles", **kwargs  # noqa: F821
    ) -> IBSKickCoefficients:
        r"""
        .. versionadded:: 0.5.0

        TODO: NEEDS A REFERENCE FOR THE IMPLEMENTATION AND A CITATION.

        .. note::
            This functionality is separate from the kick application because it internally triggers
            the computation of the analytical growth rates, and we don't necessarily want to
            recompute these at every turn. Meanwhile, the kicks **should** be applied at every turn.

        Args:
            particles (xpart.Particles): the particles to apply the IBS kicks to.
            **kwargs: any keyword arguments will be passed to ???.

        Returns:
            An `IBSKickCoefficients` object with the computed coefficients used for the kick application.
        """
        # ----------------------------------------------------------------------------------------------
        result = 1
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.coefficients = result
        return result

    def apply_ibs_kick(self, particles: "xpart.Particles", n_slices: int = 40) -> None:  # noqa: F821
        r"""
        .. versionadded:: 0.5.0

        TODO: NEEDS A REFERENCE FOR THE IMPLEMENTATION AND A CITATION.

        Args:
            particles (xpart.Particles): the `xpart.Particles` object to apply ``IBS`` kicks to.
            n_slices (int): the number of slices to use for the computation of the line density.
                Defaults to 40.

        Raises:
            AttributeError: if the ``IBS`` kick coefficients have not yet been computed.
        """
        # ----------------------------------------------------------------------------------------------
        # Check that the kick coefficients have been computed beforehand
        if self.coefficients is None:
            LOGGER.error("Attempted to apply IBS kick without having computed kick coefficients first.")
            raise AttributeError(
                "IBS kick coefficients have not been computed yet, cannot apply kick to particles.\n"
                "Please call the `compute_kick_coefficients` method first."
            )
        # ----------------------------------------------------------------------------------------------
        # TODO: implement
        pass
