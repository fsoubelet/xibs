"""
xibs package
~~~~~~~~~~~~
xibs is a prototype library for an IBS implementation in Python.
It builds on 

:copyright: (c) 2019-2020 by Felix Soubelet.
:license: MIT, see LICENSE for more details.
"""
from .analytical import Nagaitsev
from .inputs import BeamParameters, OpticsParameters
from .kinetic import KineticIBS
from .simple import SimpleIBS
from .version import VERSION

__title__ = "xibs"
__description__ = "Prototype Intra-Beam Scattering implementation for Xsuite."
__url__ = "https://github.com/fsoubelet/xibs"
__version__ = VERSION
__author__ = "Felix Soubelet"
__author_email__ = "felix.soubelet@cern.ch"
__license__ = "Apache-2.0"

# TODO: decide what to expose as top-level
__all__ = [BeamParameters, OpticsParameters, Nagaitsev, SimpleIBS, KineticIBS]
