"""
xibs package
~~~~~~~~~~~~

xibs is a prototype library for an IBS modelling implementation in Python.
It provides both an analytical and kick-based approach to IBS modelling. 

:copyright: (c) 2023 Felix Soubelet.
:license: Apache-2.0, see LICENSE file for more details.
"""
from .analytical import BjorkenMtingwaIBS, NagaitsevIBS
from .dispatch import ibs
from .inputs import BeamParameters, OpticsParameters
from .kicks import KineticKickIBS, SimpleKickIBS
from .version import VERSION

__title__ = "xibs"
__description__ = "Prototype Intra-Beam Scattering implementation for Xsuite."
__url__ = "https://github.com/fsoubelet/xibs"
__version__ = VERSION
__author__ = "Felix Soubelet"
__author_email__ = "felix.soubelet@cern.ch"
__license__ = "Apache-2.0"

# Expose chosen elements at the top-level of the package
# One can then directly import xibs.BeamParameters for instance
# Also limits what is imported when some idiot goes "from xibs import *"
__all__ = [
    "ibs",
    "BeamParameters",
    "OpticsParameters",
    "BjorkenMtingwaIBS",
    "NagaitsevIBS",
    "SimpleKickIBS",
    "KineticKickIBS",
]
