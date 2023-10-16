"""
xibs package
~~~~~~~~~~~~
xibs is a prototype library for an IBS implementation in Python.
It provides both an analytical and kick-based approach to IBS modelling. 

:copyright: (c) 2023 Felix Soubelet.
:license: Apache-2.0, see LICENSE file for more details.
"""
from .analytical import NagaitsevIBS
from .inputs import BeamParameters, OpticsParameters
from .kicks import KineticIBS, SimpleIBS
from .version import VERSION

__title__ = "xibs"
__description__ = "Prototype Intra-Beam Scattering implementation for Xsuite."
__url__ = "https://github.com/fsoubelet/xibs"
__version__ = VERSION
__author__ = "Felix Soubelet"
__author_email__ = "felix.soubelet@cern.ch"
__license__ = "Apache-2.0"

# TODO: decide what to expose as top-level
__all__ = [BeamParameters, OpticsParameters, NagaitsevIBS, SimpleIBS, KineticIBS]
