"""
.. _xibs-kinetic:

Kinetic Formalism
-----------------

Module with user-facing API to compute diffusion and friction terms according to the kinetic formalism, as well as the corresponding kinetic IBS kicks and apply them to particles.
"""
from dataclasses import dataclass

import numpy as np

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


# ----- Main class to compute kinetic coefficients and apply kicks ----- #


# It does not seem like any of the calculations in here need Nagaitsev results.
class KineticKickIBS:
    """
    A single class to compute the IBS diffusion and friction coefficients according
    to the kinetic IBS formalism of :cite:`NuclInstr:Zenkevich:Kinetic_IBS`.
    The class initiates from a `BeamParameters` and an `OpticsParameters` objects.

    Attributes:
        beam_parameters (BeamParameters): the beam parameters to use for the calculations.
        optics (OpticsParameters): the optics parameters to use for the calculations.
        diffusion_coeffs (DiffusionCoefficients): the computed diffusion coefficients. This
            self-updates when they are computed with the `diffusion_coefficients` method.
        friction_coeffs (FrictionCoefficients): the computed friction coefficients. This
            self-updates when they are computed with the `growth_rates` method.
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        self.beam_parameters: BeamParameters = beam_params
        self.optics: OpticsParameters = optics
        # These self-update when they are computed, but can be overwritten by the user
        self.diffusion_coeffs: DiffusionCoefficients = None
        self.friction_coeffs: FrictionCoefficients = None

    # TODO: go over with Michalis on his old code and determine what is being done before porting it
