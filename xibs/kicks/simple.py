"""
.. _xibs-simple:

Simple Kicks
------------

Module with user-facing API to compute simple IBS kicks from the Nagaitsev integrals and apply them to particles.
"""
import numpy as np

from xibs.analytical import NagaitsevIntegrals
from xibs.inputs import BeamParameters, OpticsParameters

# ----- Main class to compute Nagaitsev integrals and IBS growth rates ----- #


# In here we do need Nagaitsev results, so the kick method will ask for a NagaitsevIntegrals object.
class SimpleKickIBS:
    """
    A single class to compute the simple IBS kicks based on the analytical results
    obtained in `xibs.analytical` from the Nagaitsev formalism (see for reference
    :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`).
    The class initiates from a `BeamParameters` and an `OpticsParameters` objects.

    Attributes:
        beam_parameters (BeamParameters): the beam parameters to use for the calculations.
        optics (OpticsParameters): the optics parameters to use for the calculations.
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        self.beam_parameters: BeamParameters = beam_params
        self.optics: OpticsParameters = optics

    # TODO: go over with Michail on his old code and determine what is being done before porting it
