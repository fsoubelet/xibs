"""
.. _xibs-kicks:

IBS: Applying Kicks
-------------------

Module with user-facing API to compute relevant terms to IBS kicks according to different formalism: kinetic and simple.

In the kinetic formalism, the applied IBS kicks are determined from computed diffusion and friction terms.
In the simple formalism, the applied IBS kicks are determined from Nagaitsev integrals (see :ref:`xibs-analytical`).
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass
from logging import getLogger
from typing import Union

import numpy as np

from numpy.typing import ArrayLike

from xibs.analytical import AnalyticalIBS, BjorkenMtingwaIBS, IBSGrowthRates, NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters

LOGGER = getLogger(__name__)

# ----- Dataclasses to store results ----- #
# TODO: clarify the difference between these with Michalis


@dataclass
class DiffusionCoefficients:
    """Container dataclass for kinetic IBS diffusion coefficients.

    Args:
        Dx (float): horizontal diffusion coefficient in [1/s].
        Dy (float): vertical diffusion coefficient in [1/s].
        Dz (float): longitudinal diffusion coefficient in [1/s].
    """

    Dx: float
    Dy: float
    Dz: float


@dataclass
class FrictionCoefficients:
    """Container dataclass for kinetic IBS friction coefficients.

    Args:
        Fx (float): horizontal friction coefficient in [1/s].
        Fy (float): vertical friction coefficient in [1/s].
        Fz (float): longitudinal friction coefficient in [1/s].
    """

    Fx: float
    Fy: float
    Fz: float


@dataclass
class KineticCoefficients:
    """
    Container dataclass for kinetic IBS coefficients. These are computed from the diffusion
    and friction ones according to :cite:`NuclInstr:Zenkevich:Kinetic_IBS`.
    """

    Tx: float
    Ty: float
    Tz: float


# ----- Abstract Base Class to Inherit from ----- #


# TODO: adapt docstring
class KickBasedIBS(ABC):
    r"""
    .. versionadded:: 0.5.0

    Abstract base class for kick-based IBS effects, from which all
    implementations inherit.

    Attributes:
        analytical_ibs (AnalyticalIBS): an analytical IBS class to compute growth rates.
        beam_parameters (BeamParameters): the beam parameters to use for analytical IBS.
        optics (OpticsParameters): the optics parameters to use for the analytical IBS.
    """

    # TODO: make a choice on the user API (do we give a formalism, provide the rates?)
    # provide the bp and op, then each subclass hardcodes the formalism? Give choice and use
    # the dispatcher to call the one asked for by the user? Need to talk with Michalis to
    # understand the implications of Simple vs Kinetic.
    def __init__(
        self, beam_params: BeamParameters, optics: OpticsParameters, analytical_implementation: AnalyticalIBS
    ) -> None:
        self.analytical_ibs: AnalyticalIBS = analytical_implementation(beam_params, optics)
        self.beam_parameters: BeamParameters = self.analytical_ibs.beam_parameters
        self.optics: OpticsParameters = self.analytical_ibs.optics
        # These self-update when computed, but can be overwritten by the user
        self.diffusion_coefficients: DiffusionCoefficients = None
        self.friction_coefficients: FrictionCoefficients = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__} object for kick-based IBS calculations."

    def __repr__(self) -> str:
        return self.__str__()

    # TODO: remember I moved the 2 * sigma_t * sqrt(pi) term out of this function. It should be
    # computed and included by the kick application method
    def line_density(self, particles: "xpart.Particles", n_slices: int) -> ArrayLike:
        r"""
        .. versionadded:: 0.5.0

        Returns the "line density" of the `Particles` object, along its longitudinal axis, which
        corresponds to the :math:`\rho{t}` term in Eq (8) of :cite:`PRAB:Bruce:Simple_IBS_Kicks`.
        The density is used as a weight factor for the application of IBS kicks: particles in the
        denser parts of the bunch will receive a larger kick, and vice versa. See section III.C of
        the above reference details.

        .. tip::
            The calculation is done according to the following steps:

                - Gets the longitudinal coordinates of the active particles (state > 0) in the `Particles` object.
                - Determines coordinate cuts at front and back of the bunch, as well as slice width.
                - Determines bin edges and bin centers for the distribution for the chosen number of slices.
                - Computes a (normalized) histogram of the longitudinal coordinates, with the determined bins.
                - Computes and returns the line density :math:`\rho{t}`.

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
        self, particles: "xpart.Particles", **kwargs
    ) -> Union[DiffusionCoefficients, FrictionCoefficients, KineticCoefficients]:
        r"""
        .. versionadded:: 0.5.0

        TODO: fix docstring when all is set.
        Abstract method to determine the "coefficients" used in the determination of the IBS kicks.

        Args:
            particles (xpart.Particles): the particles to apply the IBS kicks to.
            **kwargs: any keyword arguments will be passed to the growth rates calculation call
                (`self.analytical_ibs.growth_rates`). Note that `epsx`, `epsy`, `sigma_delta`,
                and `bunch_length` are already provided.
        """
        raise NotImplementedError(
            "This method should be implemented in all child classes, but it hasn't been for this one."
        )

    @abstractmethod
    def apply_ibs_kick(self, particles: "xpart.Particles") -> None:
        r"""
        .. versionadded:: 0.5.0

        TODO: fix docstring when all is set.
        Abstract method to apply IBS kicks to a `xpart.Particles` object.

        Args:
            particles (xpart.Particles): the particles to apply the IBS kicks to.
        """
        raise NotImplementedError(
            "This method should be implemented in all child classes, but it hasn't been for this one."
        )


# ----- Classes to Compute and Apply IBS Kicks ----- #


# TODO: update docstring when set on interface
# From what I understand, the SimpleKicks builds on the analytical growth rates.
class SimpleKickIBS(KickBasedIBS):
    r"""
    .. versionadded:: 0.5.0

    A single class to compute the simple IBS kicks based on the analytical results obtained with
    `xibs.analytical`. The kicks are implemented according to :cite:`PRAB:Bruce:Simple_IBS_Kicks`.
    The class initiates from a `BeamParameters` and an `OpticsParameters` objects.
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        super().__init__(beam_params, optics, analytical_implementation=NagaitsevIBS)  # TODO: hard-code?
        # These self-update when computed, but can be overwritten by the user
        self.coefficients: DiffusionCoefficients = None  # TODO: check this

    # TODO: double check the return signature after clarifying all coefficients with Michalis
    def compute_kick_coefficients(self, particles: "xpart.Particles", **kwargs) -> DiffusionCoefficients:
        r"""
        .. versionadded:: 0.5.0

        TODO: NEEDS A REFERENCE FOR THE IMPLEMENTATION AND A CITATION.

        .. note::
            This functionality is separate from the kick application because it internally triggers
            the computation of the analytical growth rates, and we don't necessarily want to
            recompute these at every turn. Meanwhile, the kicks **should** be applied at every turn.

        Args:
            particles (xpart.Particles): the particles to apply the IBS kicks to.
            **kwargs: any keyword arguments will be passed to the growth rates calculation call
                (`self.analytical_ibs.growth_rates`). Note that `epsx`, `epsy`, `sigma_delta`,
                and `bunch_length` are already provided.

        Returns:
            A `DiffusionCoefficients` object with the computed diffusion coefficients.
        """
        # ----------------------------------------------------------------------------------------------
        # Compute the (geometric) emittances, momentum spread and bunch length from the Particles object
        LOGGER.debug("Computing emittances, momentum spread and bunch length from particles")
        sigma_delta: float = np.std(particles.delta[particles.state > 0])
        bunch_length: float = np.std(particles.zeta[particles.state > 0])
        sigma_x: float = np.std(particles.x[particles.state > 0])
        sigma_y: float = np.std(particles.y[particles.state > 0])
        # TODO: Why does Michalis take only the first value of d[xy] and bet[xy] in here?
        geom_epsx: float = (sigma_x**2 - (self.optics.dx[0] * sigma_delta) ** 2) / self.optics.betx[0]
        geom_epsy: float = (sigma_y**2 - (self.optics.dy[0] * sigma_delta) ** 2) / self.optics.bety[0]
        # ----------------------------------------------------------------------------------------------
        # Computing momentum - TODO: same here, why the first value??
        sigma_px_normalized: float = np.std(particles.px[particles.state > 0]) / np.sqrt(
            1 + self.optics.alfx[0] ** 2
        )
        sigma_py_normalized: float = np.std(particles.py[particles.state > 0]) / np.sqrt(
            1 + self.optics.alfy[0] ** 2
        )
        # ----------------------------------------------------------------------------------------------
        # Computing the growth rates
        growth_rates: IBSGrowthRates = self.analytical_ibs.growth_rates(
            geom_epsx, geom_epsy, sigma_delta, bunch_length, **kwargs
        )
        Tx, Ty, Tz = astuple(growth_rates)
        # TODO: figure out why Michalis did not allow negative values?
        Tx = 0 if Tx < 0 else Tx
        Ty = 0 if Ty < 0 else Ty
        Tz = 0 if Tz < 0 else Tz
        # ----------------------------------------------------------------------------------------------
        # Compute the "kicks coefficients" (DSx, DSy, DSz from Michalis' mess)
        LOGGER.debug("Computing and applying the kicks to the particles")
        DSx = sigma_px_normalized * np.sqrt(2 * Tx / self.optics.revolution_frequency)
        DSy = sigma_py_normalized * np.sqrt(2 * Ty / self.optics.revolution_frequency)
        DSz = (
            sigma_delta
            * np.sqrt(2 * Tz / self.optics.revolution_frequency)
            * self.beam_parameters.beta_rel**2
        )
        result = DiffusionCoefficients(DSx, DSy, DSz)
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.coefficients = result
        return result

    def apply_ibs_kick(self, particles: "xpart.Particles", n_slices: int = 40) -> None:
        r"""
        .. versionadded:: 0.5.0

        Compute the momentum kick to apply based on the provided `xpart.Particles` object and the
        analytical growth rates for the lattice. The kicks are implemented according to Eq (8) of
        :cite:`PRAB:Bruce:Simple_IBS_Kicks`.

        Args:
            particles (xpart.Particles): the `xpart.Particles` object to compute the line density for.
            n_slices (int): the number of slices to use for the computation of the bins.
        """
        # ----------------------------------------------------------------------------------------------
        # Check that the kick coefficients have been computed beforehand
        if self.coefficients is None:
            LOGGER.error("Attempted to apply IBS kick without having computed kick coefficients first.")
            raise ValueError(
                "IBS kick coefficients have not been computed yet, cannot apply kick to particles.\n"
                "Please call the `compute_kick_coefficients` method first."
            )
        # ----------------------------------------------------------------------------------------------
        # Compute the line density - this is the rho(t) term in Eq (8) of reference
        rho_t: np.ndarray = self.line_density(particles, n_slices)
        # ----------------------------------------------------------------------------------------------
        # Determine scaling factor, corresponding to in 2 * sigma_t * sqrt(pi) in Eq (8) of reference
        zeta: np.ndarray = particles.zeta[particles.state > 0]  # careful to only consider active particles
        bunch_length_rms: float = np.std(zeta)  # rms bunch length in [m]
        scaling_factor: float = 2 * np.pi * bunch_length_rms
        # ----------------------------------------------------------------------------------------------
        # Determining kicks to apply - this corresponds to the full result of Eq (8) of reference
        LOGGER.debug("Determining kicks to apply")
        _size_x = particles.px[particles.state > 0].shape[0]
        delta_px: np.ndarray = (
            np.random.normal(loc=0, scale=self.coefficients.Dx, size=_size_x)
            * np.sqrt(rho_t)
            * np.sqrt(scaling_factor)
        )
        _size_y = particles.py[particles.state > 0].shape[0]
        delta_py: np.ndarray = (
            np.random.normal(loc=0, scale=self.coefficients.Dy, size=_size_y)
            * np.sqrt(rho_t)
            * np.sqrt(scaling_factor)
        )
        _size_delta = particles.delta[particles.state > 0].shape[0]
        delta_delta: np.ndarray = (
            np.random.normal(loc=0, scale=self.coefficients.Dz, size=_size_delta)
            * np.sqrt(rho_t)
            * np.sqrt(scaling_factor)
        )
        # ----------------------------------------------------------------------------------------------
        # Apply the kicks to the particles
        particles.px[particles.state > 0] += delta_px
        particles.py[particles.state > 0] += delta_py
        particles.delta[particles.state > 0] += delta_delta


# TODO: update docstring when set on interface
# It does seem that Michalis for kinetic uses some of the R1, R2 etc terms from the Nagaitsev
# formalism for some reason? Will need to clarify with him.
class KineticKickIBS(KickBasedIBS):
    r"""
    .. versionadded:: 0.5.0

    A single class to compute the IBS diffusion and friction coefficients according
    to the kinetic IBS formalism of :cite:`NuclInstr:Zenkevich:Kinetic_IBS`.
    The class initiates from a `BeamParameters` and an `OpticsParameters` objects.

    Attributes:

    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        super().__init__(beam_params, optics, analytical_implementation=NagaitsevIBS)  # TODO: hard-code?
        # These self-update when computed, but can be overwritten by the user
        self.coefficients: DiffusionCoefficients = None  # TODO: check this

    def compute_kick_coefficients(self, particles: "xpart.Particles", **kwargs) -> DiffusionCoefficients:
        r"""
        .. versionadded:: 0.5.0

        TODO: NEEDS A REFERENCE FOR THE IMPLEMENTATION AND A CITATION.

        .. note::
            This functionality is separate from the kick application because it internally triggers
            the computation of the analytical growth rates, and we don't necessarily want to
            recompute these at every turn. Meanwhile, the kicks **should** be applied at every turn.

        Args:
            particles (xpart.Particles): the particles to apply the IBS kicks to.
            **kwargs: any keyword arguments will be passed to the growth rates calculation call
                (`self.analytical_ibs.growth_rates`). Note that `epsx`, `epsy`, `sigma_delta`,
                and `bunch_length` are already provided. TODO: check we actually do that.

        Returns:
            A ??? object with the computed diffusion coefficients.
        """
        # ----------------------------------------------------------------------------------------------
        result = 1
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.coefficients = result
        return result

    def apply_ibs_kick(self, particles: "xpart.Particles", n_slices: int = 40) -> None:
        r"""
        .. versionadded:: 0.5.0

        TODO: NEEDS A REFERENCE FOR THE IMPLEMENTATION AND A CITATION.

        Args:
            particles (xpart.Particles): the particles to apply the IBS kicks to.
        """
        # ----------------------------------------------------------------------------------------------
        # Determine scaling factor corresponding to in 2 * sigma_t * sqrt(pi) Eq (8) of reference
        zeta: np.ndarray = particles.zeta[particles.state > 0]  # careful to only consider active particles
        bunch_length_rms: float = np.std(zeta)  # rms bunch length in [m]
        scaling_factor = 2 * np.pi * bunch_length_rms
        # ----------------------------------------------------------------------------------------------
        # TODO: implement the rest
        pass
