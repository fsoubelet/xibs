"""
.. _xibs-kinetic:

Kinetic Formalism
-----------------

Module with user-facing API to compute diffusion and friction terms from the Nagaitsev integrals according to the kinetic formalism, as well as the corresponding kinetic IBS kicks and apply them to particles.
"""
from dataclasses import dataclass

import numpy as np

from xibs.analytical import BeamParameters, OpticsParameters

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


# ----- Main class to compute Nagaitsev integrals and IBS growth rates ----- #


class KineticIBS:
    pass
