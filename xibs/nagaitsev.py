"""
.. _xibs-nagaitsev:

Nagaitsev Integrals
-------------------

Module with functionality to perform analytical calculations according to Nagaitsev's formalism.
A user-facing class is provided which computes the Nagaitsev integrals based on beam optics.
The formalism from which formulas and calculations are implemented can be found in :cite:p:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

from dataclasses import InitVar, dataclass, field

import numpy as np

from numpy.typing import ArrayLike
from scipy.constants import c, hbar
from scipy.integrate import quad
from scipy.interpolate import interp1d

# ----- Dataclasses for inputs to Nagaitsev class ----- #


@dataclass
class BeamParameters:
    """Container dataclass for necessary beam parameters. It is initiated from
    the `xpart.Particles` object to track in your line with ``xsuite``.

    Args:
        particles (xpart.Particles): the generates particles to be tracked or used
            in the line.

    Attributes:
        n_part (int): number of simulated particles.
        particle_charge (int): elementary particle charge, in # of Coulomb charges (for
            instance 1 for electron or proton).
        particle_mass_GeV (float): particle mass in [GeV].
        total_energy_GeV (float): total energy of the simulated particles in [GeV].
        gamma_rel (float): relativistic gamma of the simulated particles.
        beta_rel (float): relativistic beta of the simulated particles.
        particle_classical_radius_m (float): the particles' classical radius in [m].
    """

    # ----- To be provided at initialization ----- #
    particles: InitVar["xpart.Particles"]  # Almost all is derived from there, this is not kept!
    # ----- Below are attributes derived from the Particles object ----- #
    # The following are Npart, Ncharg, E_rest, EnTot, gammar, betar and c_rad in Michail's code
    n_part: int = field(init=False)
    particle_charge: int = field(init=False)
    particle_mass_GeV: float = field(init=False)
    total_energy_GeV: float = field(init=False)
    gamma_rel: float = field(init=False)
    beta_rel: float = field(init=False)
    particle_classical_radius_m: float = field(init=False)

    def __post_init__(self, particles: "xpart.Particles"):
        # Attributes derived from the Particles object
        self.n_part = particles.weight[0] * particles.gamma0.shape[0]
        self.particle_charge = particles.q0
        self.particle_mass_GeV = particles.mass0 * 1e-9
        self.total_energy_GeV = np.sqrt(particles.p0c[0] ** 2 + particles.mass0**2) * 1e-9
        self.gamma_rel = particles.gamma0[0]
        self.beta_rel = particles.beta0[0]
        self.particle_classical_radius_m = particles.get_classical_particle_radius0()


@dataclass
class OpticsParameters:
    """Container dataclass for necessary optics parameters. It is initiated from
    the results of a ``TWISS`` command with an ``xsuite``.

    Args:
        twiss (xtrack.twiss.TwissTable): the resulting table of a ``TWISS`` call
            on the line in ``xsuite``.
        revolution_frequency (float): revolution frequency of the machine in [Hz].
            If initiating after from ``xsuite`` elements, this can be obtained with
            `particle_ref.beta0[0] * scipy.constants.c / twiss.s[-1]`.

    Attributes:
        revolution_frequency (float): revolution frequency of the machine in [Hz].
        s (ArrayLike): longitudinal positions of the machine elements in [m].
        circumference (float): machine circumference in [m].
        slip_factor (float): slip factor of the machine.
        betx (ArrayLike): horizontal beta functions in [m].
        bety (ArrayLike): vertical beta functions in [m].
        alfx (ArrayLike): horizontal alpha functions.
        alfy (ArrayLike): vertical alpha functions.
        dx (ArrayLike): horizontal dispersion functions in [m].
        dy (ArrayLike): vertical dispersion functions in [m].
        dpx (ArrayLike): horizontal dispersion (d px / d delta).
        dpy (ArrayLike): horizontal dispersion (d px / d delta).
    """

    # ----- To be provided at initialization ----- #
    twiss: InitVar["xtrack.twiss.TwissTable"]  # Almost all is derived from there, this is not kept!
    revolution_frequency: float
    # ----- Below are attributes derived from the twiss table ----- #
    s: ArrayLike = field(init=False)
    circumference: float = field(init=False)
    slip_factor: float = field(init=False)
    betx: ArrayLike = field(init=False)
    bety: ArrayLike = field(init=False)
    alfx: ArrayLike = field(init=False)
    alfy: ArrayLike = field(init=False)
    # The following four are eta_x, eta_y, eta_dx and eta_dy in michail's code
    dx: ArrayLike = field(init=False)
    dy: ArrayLike = field(init=False)
    dpx: ArrayLike = field(init=False)
    dpy: ArrayLike = field(init=False)
    # Below are ONLY USED in the CoulogConst function -> extract? Kept private for now
    _bx_bar: ArrayLike = field(init=False)
    _by_bar: ArrayLike = field(init=False)
    _dx_bar: ArrayLike = field(init=False)
    _dy_bar: ArrayLike = field(init=False)

    def __post_init__(self, twiss: "xtrack.twiss.TwissTable"):
        # Attributes derived from the TwissTable
        self.s = twiss.s
        self.circumference = twiss.s[-1]
        self.slip_factor = twiss["slip_factor"]
        self.betx = twiss.betx
        self.bety = twiss.bety
        self.alfx = twiss.alfx
        self.alfy = twiss.alfy
        self.dx = twiss.dx
        self.dy = twiss.dy
        self.dpx = twiss.dpx
        self.dpy = twiss.dpy
        # Interpolated beta and dispersion functions for the calculation below
        _bxb = interp1d(self.s, self.betx)
        _byb = interp1d(self.s, self.bety)
        _dxb = interp1d(self.s, self.dx)
        _dyb = interp1d(self.s, self.dy)
        # Computing "average" of these functions - better here than a simple np.mean
        # calculation because the latter doesn't take in consideration element lengths
        self._bx_bar = quad(_bxb, self.s[0], self.s[-1])[0] / self.circumference
        self._by_bar = quad(_byb, self.s[0], self.s[-1])[0] / self.circumference
        self._dx_bar = quad(_dxb, self.s[0], self.s[-1])[0] / self.circumference
        self._dy_bar = quad(_dyb, self.s[0], self.s[-1])[0] / self.circumference


# ----- Dataclasses to store results ----- #


@dataclass
class NagaitsevIntegrals:
    """Container dataclass for Nagaitsev integrals results.

    Args:
        Ix (float): horizontal Nagaitsev integral.
        Iy (float): vertical Nagaitsev integral.
        Iz (float): longitudinal Nagaitsev integral.
    """

    Ix: float
    Iy: float
    Iz: float


@dataclass
class IBSGrowthRates:
    """Container dataclass for IBS growth rates results.

    Args:
        Tx (float): horizontal IBS growth rate.
        Ty (float): vertical IBS growth rate.
        Tz (float): longitudinal IBS growth rate.
    """

    Tx: float
    Ty: float
    Tz: float


# ----- Main class to compute Nagaitsev integrals and IBS growth rates ----- #


class Nagaitsev:
    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        self.beam_parameters = beam_params
        self.optics = optics

    def _coulomb_log_constant(
        self, geom_epsx: float, geom_epxy: float, sigma_delta: float, bunch_length: float
    ):
        """
        This is the full constant factor (building on Coulomb Log (constant) from Eq (9) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`).
        Calculates Coulog constant, then log and returns it multiplied by sthe rest of the constant term in the equation.
        Michail might have gotten this from MAD-X?

        For a good resource on the Coulomb Log, see: https://docs.plasmapy.org/en/stable/notebooks/formulary/coulomb.html

        Args:
            epsx (float): horizontal geometric emittance in [m].
            epxy (float): vertical geometric emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): bunch length in [m].
        """
        # TODO: figure this all out by finding source (MAD-X IBS?), give proper variable names and document the calculation in docstring.
        # fmt: off
        Etrans = (  # who the fuck are you?
            5e8 * (self.beam_parameters.gamma_rel * self.beam_parameters.total_energy_GeV - self.beam_parameters.particle_mass_GeV) * (geom_epsx / self.optics._bx_bar)
        )
        # fmt: on
        TempeV = 2.0 * Etrans
        sigxcm = 100 * np.sqrt(geom_epsx * self.optics._bx_bar + (self.optics._dx_bar * sigma_delta) ** 2)
        sigycm = 100 * np.sqrt(geom_epxy * self.optics._by_bar + (self.optics._dy_bar * sigma_delta) ** 2)
        sigtcm = 100 * bunch_length
        volume = 8.0 * np.sqrt(np.pi**3) * sigxcm * sigycm * sigtcm
        densty = self.beam_parameters.n_part / volume
        debyul = 743.4 * np.sqrt(TempeV / densty) / self.beam_parameters.particle_charge
        rmincl = 1.44e-7 * self.beam_parameters.particle_charge**2 / TempeV
        rminqm = hbar * c * 1e5 / (2.0 * np.sqrt(2e-3 * Etrans * self.beam_parameters.particle_mass_GeV))
        rmin = max(rmincl, rminqm)
        rmax = min(sigxcm, debyul)
        coulog = np.log(rmax / rmin)
        # fmt: off
        Ncon = (
            self.beam_parameters.n_part
            * self.beam_parameters.particle_classical_radius_m**2
            * c 
            / (12 * np.pi * self.beam_parameters.beta_rel**3 * self.beam_parameters.gamma_rel**5 * bunch_length)
        )
        # fmt: on
        return Ncon * coulog
