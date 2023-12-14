"""
.. _xibs-kicks:

IBS: Applying Kicks
-------------------

Module with user-facing API to compute relevant terms to IBS kicks according to different formalism: kinetic and simple.

In the kinetic formalism, the applied IBS kicks are determined from computed diffusion and friction terms.
In the simple formalism, the applied IBS kicks are determined from Nagaitsev integrals (see :ref:`xibs-analytical`).
"""
from dataclasses import dataclass

import numpy as np

from xibs.analytical import NagaitsevIntegrals
from xibs.inputs import BeamParameters, OpticsParameters

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


# ----- Classes to Compute and Apply IBS Kicks ----- #


# In here we do need Nagaitsev results, so the kick method will ask for a NagaitsevIntegrals object.
class SimpleKickIBS:
    r"""
    .. versionadded:: 0.4.0

    A single class to compute the simple IBS kicks based on the analytical results obtained with
    `xibs.analytical`. The kicks are implemented according to :cite:`PRAB:Bruce:Simple_IBS_Kicks`.
    The class initiates from a `BeamParameters` and an `OpticsParameters` objects.

    Attributes:
        beam_parameters (BeamParameters): the beam parameters to use for the calculations.
        optics (OpticsParameters): the optics parameters to use for the calculations.
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        self.beam_parameters: BeamParameters = beam_params
        self.optics: OpticsParameters = optics

    # TODO: go over with Michalis on his old code and determine what is being done before porting it


# It does not seem like any of the calculations in here need Nagaitsev results.
class KineticKickIBS:
    r"""
    .. versionadded:: 0.4.0

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
        self.beam_parameters: BeamParameters = beam_params
        self.optics: OpticsParameters = optics
        # These self-update when they are computed, but can be overwritten by the user
        self.diffusion_coeffs: DiffusionCoefficients = None
        self.friction_coeffs: FrictionCoefficients = None
        self.kinectic_coeffs: KineticCoefficients = None

    # TODO: go over with Michalis on his old code and determine what is being done before porting it
