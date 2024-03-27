"""
Tests in here check that errors that should be raised by the analytical classes
(BjorkenMtingwaIBS and NagaitsevIBS) are indeed raised.
"""
import pytest

from xibs.analytical import BjorkenMtingwaIBS, NagaitsevIBS
from xibs.inputs import BeamParameters, OpticsParameters
