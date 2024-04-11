"""
.. _xibs-dispatch:

Formalism Dispatch
------------------

This module provides a single function, `ibs`, to ease the class instantiation for the user. The function returns an
instance of the relevant IBS modelling class based on the desired formalism.
"""
from logging import getLogger
from typing import Union

from xibs.analytical import BjorkenMtingwaIBS, NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS, SimpleKickIBS

LOGGER = getLogger(__name__)


def ibs(
    beam_params: BeamParameters, optics_params: OpticsParameters, formalism: str
) -> Union[BjorkenMtingwaIBS, NagaitsevIBS, KineticKickIBS, SimpleKickIBS]:
    """A dispatch function to return the appropriate IBS modelling class based on the desired formalism.

    Args:
        beam_parameters (BeamParameters): the beam parameters to use for the calculations. They will
            be used to initialize the relevant IBS class.
        optics_params (OpticsParameters): the optics parameters to use for the calculations. They will be
            used to initialize the relevant IBS class.
        formalism (str): the desired IBS modelling formalism. This determines which IBS class will be
            returned. Valid options are: "bjorken-mtingwa" (or also "b&m"), "nagaitsev", "kinetic", and
            "simple". This argument is case-insensitive. See the return annotation below for more details.

    Returns:
        An instance of the relevant IBS modelling class. The chosen class is determined by the `formalism`
        argument. Below are the various options and corresponding returned classes:

        - **bjorken-mtingwa** or **b&m**: returns a `BjorkenMtingwaIBS` instance (see :ref:`xibs-analytical`),
          which implements the ``MAD-X`` analytical calculations for IBS growth rates that build on Bjorken &
          Mtingwa's formalism to take in consideration vertical dispersion.
        - **nagaitsev**: returns a `NagaitsevIBS` instance (see :ref:`xibs-analytical`), which implements the
          Nagaitsev analytical calculation for IBS growth rates. It is very fast, but does not take in
          consideration the vertical dispersion.
        - **kinetic**: returns a `KineticKickIBS` instance (see :ref:`xibs-kicks`), which implements the
          kinetic kick-based formalism.
        - **simple**: returns a `SimpleKickIBS` instance (see :ref:`xibs-kicks`), which implements the kick
          formalism from Nagaitsev analytical values.

    Example:
        .. code-block:: python

            from xibs import ibs, BeamParameters, OpticsParameters

            # Here is where you would define your inputs
            beam_params = BeamParameters(...)
            optics = OpticsParameters(...)

            # Get the proper modelling class based on the demanded formalism
            BM_IBS = ibs(beam_params, optics, formalism="b&m")  # a BjorkenMtingwaIBS instance
            NAGAITSEV_IBS = ibs(beam_params, optics, formalism="nagaitsev")  # a NagaitsevIBS instance
            KINETIC_IBS = ibs(beam_params, optics, formalism="kinetic")  # a KineticKickIBS instance
            SIMPLE_IBS = ibs(beam_params, optics, formalism="simple")  # a SimpleKickIBS instance
    """
    # ----------------------------------------------------------------------------------------------
    # Check the validity of the 'formalism' argument
    if formalism.lower() not in ("bjorken-mtingwa", "b&m", "nagaitsev", "kinetic", "simple"):
        LOGGER.error(f"Invalid formalism '{formalism}' demanded.")
        raise ValueError(
            f"Unknown formalism '{formalism}' demanded. The valid options are (case-insensitive): "
            "'bjorken-mtingwa' (or 'b&m'), 'nagaitsev', 'kinetic', and 'simple'."
        )
    # ----------------------------------------------------------------------------------------------
    # Dispatch the appropriate class based on the 'formalism' argument
    # This can be a pattern matching statement when Python 3.10 becomes our lowest supported version
    if formalism.lower() in ("bjorken-mtingwa", "b&m"):
        return BjorkenMtingwaIBS(beam_params, optics_params)
    if formalism.lower() == "nagaitsev":
        return NagaitsevIBS(beam_params, optics_params)
    if formalism.lower() == "kinetic":
        return KineticKickIBS(beam_params, optics_params)
    return SimpleKickIBS(beam_params, optics_params)
