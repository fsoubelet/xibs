"""

.. _demo-kinetic:

==========================
Kinetic IBS Kicks Tracking
==========================

This example shows how to use the `~.xibs.kicks.kinetic.KineticKickIBS` class
to calculate IBS kicks to apply to an `xpart.Particles` object during tracking.

We will demonstrate using an `xtrack.Line` of the ``MACHINE``,
for a PARTICLE beam.

.. important::
    As `xibs` is still a prototype implementation, the integration of IBS effects
    into `xsuite` tracking simulations is not seamless, as we will see.
"""
# sphinx_gallery_thumbnail_number = 1
import json
import warnings

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import xpart as xp
import xtrack as xt

from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks.kinetic import KineticKickIBS

###############################################################################
# Let's start by defining the line and particle information, as well as some
# parameters for later use:

line_file = "lines/machine_file.json"
harmonic_number = 2852
rf_voltage = 4.5  # in MV
energy_loss = 0  # let's pretend
bunch_intensity = 4.4e9
sigma_z = 1.58e-3
nemitt_x = 5.6644e-07
nemitt_y = 3.7033e-09
n_part = int(5e3)


#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
