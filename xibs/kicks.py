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
from dataclasses import dataclass
from logging import getLogger

import numpy as np

from numpy.typing import ArrayLike

from xibs.analytical import AnalyticalIBS, IBSGrowthRates
from xibs.dispatch import ibs
from xibs.inputs import BeamParameters, OpticsParameters

LOGGER = getLogger(__name__)

# ----- Dataclasses to store results ----- #


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
    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters, formalism: str) -> None:
        self.analytical_ibs: AnalyticalIBS = ibs(beam_params, optics, formalism)
        self.beam_parameters: BeamParameters = self.analytical_ibs.beam_parameters
        self.optics: OpticsParameters = self.analytical_ibs.optics
        # These self-update when computed, but can be overwritten by the user
        self.diffusion_coefficients: DiffusionCoefficients = None
        self.friction_coefficients: FrictionCoefficients = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__} object for kick-based IBS calculations.\n"

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

    # TODO: implement signature and leave up to inherited class?
    @abstractmethod
    def apply_ibs_kick(self, particles: "xpart.Particles") -> None:
        r"""
        .. versionadded:: 0.5.0

        Abstract method to apply IBS kicks to a `xpart.Particles` object.

        Args:
            particles (xpart.Particles): the particles to apply the IBS kicks to.
        """
        # TODO: remember I will move the 2 * sigma_t * sqrt(pi) term out of line-density and it should be
        # computed here instead
        raise NotImplementedError(
            "This method should be implemented in all child classes, but it hasn't been for this one."
        )


# ----- Classes to Compute and Apply IBS Kicks ----- #


# In here we do need Nagaitsev results, so the kick method will trigger computing NagaitsevIntegrals?
class SimpleKickIBS(KickBasedIBS):
    r"""
    .. versionadded:: 0.5.0

    A single class to compute the simple IBS kicks based on the analytical results obtained with
    `xibs.analytical`. The kicks are implemented according to :cite:`PRAB:Bruce:Simple_IBS_Kicks`.
    The class initiates from a `BeamParameters` and an `OpticsParameters` objects.

    Attributes:
        beam_parameters (BeamParameters): the beam parameters to use for the calculations.
        optics (OpticsParameters): the optics parameters to use for the calculations.
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        super().__init__(beam_params, optics, formalism="nagaitsev")  # TODO: hard-code?


# It does not seem like any of the calculations in here need Nagaitsev results. Could use B&M?
class KineticKickIBS(KickBasedIBS):
    r"""
    .. versionadded:: 0.5.0

    A single class to compute the IBS diffusion and friction coefficients according
    to the kinetic IBS formalism of :cite:`NuclInstr:Zenkevich:Kinetic_IBS`.
    The class initiates from a `BeamParameters` and an `OpticsParameters` objects.

    Attributes:
        beam_parameters (BeamParameters): the beam parameters to use for the calculations.
        optics (OpticsParameters): the optics parameters to use for the calculations.
        diffusion_coeffs (DiffusionCoefficients): the computed diffusion coefficients. This
            self-updates when they are computed with the `kinetic_coefficients` method.
        friction_coeffs (FrictionCoefficients): the computed friction coefficients. This
            self-updates when they are computed with the `kinetic_coefficients` method.
        kinectic_coeffs (KineticCoefficients): the computed kinetic coefficients from which
            the kinetic kicks are determined. This self-updates when they are computed with
            the `kinetic_coefficients` method?
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        super().__init__(beam_params, optics, formalism="nagaitsev")  # TODO: hard-code?
