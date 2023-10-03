"""
.. _xibs-nagaitsev:

Nagaitsev Integrals
-------------------

Module with functionality to perform analytical calculations according to Nagaitsev's formalism.
A user-facing class is provided which computes the Nagaitsev integrals based on beam optics.
The formalism from which formulas and calculations are implemented can be found in :cite:p:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

import logging

from dataclasses import InitVar, astuple, dataclass, field

import numpy as np

from numpy.typing import ArrayLike
from scipy.constants import c, hbar
from scipy.integrate import quad
from scipy.interpolate import interp1d

from xibs.formulary import phi

LOGGER = logging.getLogger(__name__)

# ----- Dataclasses for inputs to Nagaitsev class ----- #


@dataclass
class BeamParameters:
    """Container dataclass for necessary beam parameters. It is initiated from
    the `xpart.Particles` object to track in your line with ``xsuite``.

    Args:
        particles (xpart.Particles): the generated particles to be tracked or used
            in the line. This is an init-only parameter used for instanciation and
            it will not be kept in the instance's attributes.

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
        LOGGER.debug("Initializing BeamParameters from Particles object")
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
            on the line in ``xsuite``. This is an init-only parameter used for
            instanciation and it will not be kept in the instance's attributes.
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
        LOGGER.debug("Initializing OpticsParameters from TwissTable object")
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
        # TODO: this is only needed in the coulomb logarithm calculation, could be moved there?
        # This way users don't have to see the scipy.integrate.quad warnings when instantiating
        LOGGER.debug("Interpolating beta and dispersion functions")
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
    """
    A single class to compute Nagaitsev integrals (see
    :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`)
    and IBS growth rates. It initiates from a `BeamParameters` and an `OpticsParameters` objects.

    Attributes:
        beam_parameters (BeamParameters): the beam parameters to use for the calculations.
        optics (OpticsParameters): the optics parameters to use for the calculations.
        elliptic_integrals (NagaitsevIntegrals): the computed elliptic integrals. This
            self-updates when they are computed with the `integrals` method.
        growth_rates (IBSGrowthRates): the computed IBS growth rates. This self-updates
            when they are computed with the `growth_rates` method.
    """

    def __init__(self, beam_params: BeamParameters, optics: OpticsParameters) -> None:
        self.beam_parameters: BeamParameters = beam_params
        self.optics: OpticsParameters = optics
        # These self-update when they are computed, but can be overwritten by the user
        self.elliptic_integrals: NagaitsevIntegrals = None
        self.growth_rates: IBSGrowthRates = None

    def coulomb_log(
        self, geom_epsx: float, geom_epxy: float, sigma_delta: float, bunch_length: float
    ) -> float:
        r"""
        Calculates the Coulomb logarithm based on the beam parameters and optics the class
        was initiated with. For a good introductory resource on the Coulomb Log, see:
        https://docs.plasmapy.org/en/stable/notebooks/formulary/coulomb.html

        .. note::
            This is a copy-paste of the body from `CoulogConst` in Michail's code,
            but stopping and returning as soon as the Coulomb logarithm is calculated.
            The caller function is left to compute the rest of the constant term in
            Eq (9) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.

        .. todo::
            Figure out what the hell is going on in this function and document it properly.
            Michail thinks he might have gotten it from the ``MAD-X`` ``IBS`` module?

        Args:
            epsx (float): horizontal geometric emittance in [m].
            epxy (float): vertical geometric emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): bunch length in [m].

        Returns:
            The dimensionless Coulomb logarithm :math:`\ln \left( Λ \right)`.
        """
        LOGGER.debug("Computing Coulomb logarithm for definded beam and optics parameters")
        # fmt: off
        Etrans = (  # who the fuck are you?
            5e8 * (self.beam_parameters.gamma_rel * self.beam_parameters.total_energy_GeV - self.beam_parameters.particle_mass_GeV) * (geom_epsx / self.optics._bx_bar)
        )
        # fmt: on
        TempeV = 2.0 * Etrans

        # TODO: computing sigmas, why?
        sigma_x_cm = 100 * np.sqrt(geom_epsx * self.optics._bx_bar + (self.optics._dx_bar * sigma_delta) ** 2)
        sigma_y_cm = 100 * np.sqrt(geom_epxy * self.optics._by_bar + (self.optics._dy_bar * sigma_delta) ** 2)
        sigma_t_cm = 100 * bunch_length

        # TODO: computing volume, density and maybe Debye length?
        volume = 8.0 * np.sqrt(np.pi**3) * sigma_x_cm * sigma_y_cm * sigma_t_cm
        density = self.beam_parameters.n_part / volume
        debyul = 743.4 * np.sqrt(TempeV / density) / self.beam_parameters.particle_charge  # Debye length?

        rmincl = 1.44e-7 * self.beam_parameters.particle_charge**2 / TempeV
        rminqm = hbar * c * 1e5 / (2.0 * np.sqrt(2e-3 * Etrans * self.beam_parameters.particle_mass_GeV))

        # Now compute the impact parameters and Coulomb logarithm
        bmin = max(rmincl, rminqm)
        bmax = min(sigma_x_cm, debyul)
        return np.log(bmax / bmin)

    def coulomb_log_full_constant(
        self, geom_epsx: float, geom_epxy: float, sigma_delta: float, bunch_length: float
    ) -> float:
        """
        This is the full constant factor (building on Coulomb log, see the `coulomb_log` method)
        from Eq (9) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`. This returns
        the same as what is returned by the `CoulogConst` method in Michail's code.

        .. todo::
            Check the calculation for the rest of the constant term, the expression
            in Eq (9) seems to have been massaged quite a bit.

        Args:
            epsx (float): horizontal geometric emittance in [m].
            epxy (float): vertical geometric emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): bunch length in [m].

        Returns:
            The dimensionless Coulomb logarithm :math:`\ln\\left(Λ\\right)` multiplied by the rest
            of the constant term in Eq (9) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.
        """
        # First get the coulomb logarithm from the dedicated method
        coulomb_logarithm = self.coulomb_log(geom_epsx, geom_epxy, sigma_delta, bunch_length)
        # Then the rest of the constant term in the equation
        # fmt: off
        Ncon = (
            self.beam_parameters.n_part
            * self.beam_parameters.particle_classical_radius_m**2
            * c 
            / (12 * np.pi * self.beam_parameters.beta_rel**3 * self.beam_parameters.gamma_rel**5 * bunch_length)
        )
        # fmt: on
        return Ncon * coulomb_logarithm

    def iterative_RD(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        r"""Computes the terms inside the elliptic integral in Eq (4) of
        :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.

        This is an iterative method implementation that was found by Michail (in ``C++``
        then adapted). The implementation is found in ref [5] (uses ref [4] too) of the
        same paper: :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.

        .. note::
            This is for now a copy-paste of the `RDiter` method in Michail's code.
            Some PowerPoints from Michail in an old ABP group meeting mention how this
            calculation works. Can look into this for details and documentation.

        .. todo::
            This is the most time-consuming part of the class's integrals computing. For
            optimization, since this doesn't call any internal attributes of the class
            it could be moved out (into `xibs.formulary`?) and potentially JIT-compiled with
            ``numba``. Then we import directly from the right module and call it in the
            integrals calculations.

        .. todo::
            Go through the old scripts in debugging mode and inspect what is passed in
            and then out to get a better idea of the function signature.

        Args:
            x (ArrayLike): the :math:`\lambda_1` values in Nagaitsev paper? Eigen values of
                the :math:`\bf{A}` matrix in Eq (2) which comes from B&M (ref ?). In B&M
                it is :math:`\bf{L}` matrix (ref?). This is an array with the value for each
                element in the lattice.
            y (ArrayLike): the :math:`\lambda_2` values in Nagaitsev paper? Eigen values of
                the :math:`\bf{A}` matrix in Eq (2) which comes from B&M (ref ?). In B&M
                it is :math:`\bf{L}` matrix (ref?). This is an array with the value for each
                element in the lattice.
            z (ArrayLike): the :math:`\lambda_3` values in Nagaitsev paper? Eigen values of
                the :math:`\bf{A}` matrix in Eq (2) which comes from B&M (ref ?). In B&M
                it is :math:`\bf{L}` matrix (ref?). This is an array with the value for each
                element in the lattice.

        Returns:
            An array with the result of the calculation for each element in the lattice. This
            is NOT the elliptic integral yet, it has to be integrated afterwards.
        """
        LOGGER.debug("Iteratively computing elliptic integral RD term")
        R = []
        for i, j, k in zip(x, y, z):
            x0 = i
            y0 = j
            z0 = k
            if (x0 < 0) and (y0 <= 0) and (z0 <= 0):
                print("Elliptic Integral Calculation Failed. Wrong input values!")
                return
            x = x0
            y = y0
            z = [z0]
            li = []
            Sn = []
            differ = 10e-4
            for n in range(0, 1000):
                xi = x
                yi = y
                li.append(np.sqrt(xi * yi) + np.sqrt(xi * z[n]) + np.sqrt(yi * z[n]))
                x = (xi + li[n]) / 4.0
                y = (yi + li[n]) / 4.0
                z.append((z[n] + li[n]) / 4.0)
                if (
                    (abs(x - xi) / x0 < differ)
                    and (abs(y - yi) / y0 < differ)
                    and (abs(z[n] - z[n + 1]) / z0 < differ)
                ):
                    break
            lim = n
            mi = (xi + yi + 3 * z[lim]) / 5.0
            Cx = 1 - (xi / mi)
            Cy = 1 - (yi / mi)
            Cz = 1 - (z[n] / mi)
            En = max(Cx, Cy, Cz)
            if En >= 1:
                print("Something went wrong with En")
                return
            summ = 0
            for m in range(2, 6):
                Sn.append((Cx**m + Cy**m + 3 * Cz**m) / (2 * m))
            for m in range(0, lim):
                summ += 1 / (np.sqrt(z[m]) * (z[m] + li[m]) * 4**m)

            # Ern = 3 * En**6 / (1 - En) ** (3 / 2.0)
            rn = -Sn[2 - 2] ** 3 / 10.0 + 3 * Sn[3 - 2] ** 2 / 10.0 + 3 * Sn[2 - 2] * Sn[4 - 2] / 5.0
            R.append(
                3 * summ
                + (
                    1
                    + 3 * Sn[2 - 2] / 7.0
                    + Sn[3 - 2] / 3.0
                    + 3 * Sn[2 - 2] ** 2 / 22.0
                    + 3 * Sn[4 - 2] / 11.0
                    + 3 * Sn[2 - 2] * Sn[3 - 2] / 13.0
                    + 3 * Sn[5 - 2] / 13.0
                    + rn
                )
                / (4**lim * mi ** (3 / 2.0))
            )
        # This returns an array with one value per element in the lattice
        # This is NOT the elliptic integral yet, it has to be integrated afterwards. It is the term in the integral in Eq (4) in Nagaitsev paper.
        return R

    # This is 'Nagaitsev_Integrals' from Michail's old code but it stops a bit earlier and really returns the integrals
    # The arguments used to be named Emit_x, Emit_y, Sig_M, BunchL there
    def integrals(
        self,
        geom_epsx: float,
        geom_epsy: float,
        sigma_delta: float,
    ) -> NagaitsevIntegrals:
        r"""Computes the Nagaitsev integrals, named :math:`I_x, I_y` and :math:`I_z` in this code base.

        These correspond to the integrals inside of Eq (32), (31) and (30) in
        :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`, respectively.
        The instance attribute `self.elliptic_integrals` is automatically updated
        with the results of this method. It is used for other calculations.

        .. tip::
            The calculation is done according to the following steps, which are related to different
            equations in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`:

                - Computes various intermediate terms and then :math:`a_x, a_y, a_s, a_1` and :math:`a_2` constants from Eq (18-21).
                - Computes the eigenvalues :math:`\lambda_1, \lambda_2` of the :math:`\bf{A}` matrix (:math:`\bf{L}` matrix in B&M) from Eq (22-24).
                - Iteratively computes the :math:`R_1, R_2` and :math:`R_3` terms from Eq (25-27) with the forms of Eq (5-6).
                - Computes the :math:`S_p, S_x` and :math:`S_{xp}` terms from Eq (33-35).
                - Computes and return the integrals terms in Eq (30-32).

        Args:
            epsx (float): horizontal geometric emittance in [m].
            epxy (float): vertical geometric emittance in [m].
            sigma_delta (float): momentum spread.

        Returns:
            A `NagaitsevIntegrals` object with the computed integrals for each plane.
        """
        LOGGER.info("Computing Nagaitsev integrals for defined beam and optics parameters")
        # fmt: off
        # All of the following (when type annotated as np.ndarray), hold one value per element in the lattice
        # ----------------------------------------------------------------------------------------------
        # Computing necessary intermediate terms for the following lines
        sigx: np.ndarray = np.sqrt(self.optics.betx * geom_epsx + (self.optics.dx * sigma_delta) ** 2)
        sigy: np.ndarray = np.sqrt(self.optics.bety * geom_epsy + (self.optics.dy * sigma_delta) ** 2)
        phix: np.ndarray = phi(self.optics.betx, self.optics.alfx, self.optics.dx, self.optics.dpx)
        # Computing the constants from Eq (18-21) in Nagaitsev paper
        ax: np.ndarray = self.optics.betx / geom_epsx
        ay: np.ndarray = self.optics.bety / geom_epsy
        a_s: np.ndarray = ax * (self.optics.dx**2 / self.optics.betx**2 + phix**2) + 1 / sigma_delta**2
        a1: np.ndarray = (ax + self.beam_parameters.gamma_rel**2 * a_s) / 2.0
        a2: np.ndarray = (ax - self.beam_parameters.gamma_rel**2 * a_s) / 2.0
        sqrt_term = np.sqrt(a2**2 + self.beam_parameters.gamma_rel**2 * ax**2 * phix**2)  # square root term in Eq (22-23) and Eq (33-35)
        # ----------------------------------------------------------------------------------------------
        # These are from Eq (22-24) in Nagaitsev paper, eigen values of A matrix (L matrix in B&M)
        lambda_1: np.ndarray = ay
        lambda_2: np.ndarray = a1 + sqrt_term
        lambda_3: np.ndarray = a1 - sqrt_term
        # ----------------------------------------------------------------------------------------------
        # These are the R_D terms to compute, from Eq (25-27) in Nagaitsev paper
        R1: np.ndarray = self.iterative_RD(1 / lambda_2, 1 / lambda_3, 1 / lambda_1) / lambda_1
        R2: np.ndarray = self.iterative_RD(1 / lambda_3, 1 / lambda_1, 1 / lambda_2) / lambda_2
        R3: np.ndarray = 3 * np.sqrt(lambda_1 * lambda_2 / lambda_3) - lambda_1 * R1 / lambda_3 - lambda_2 * R2 / lambda_3
        # ----------------------------------------------------------------------------------------------
        # This are the terms from Eq (33-35) in Nagaitsev paper
        Sp: np.ndarray = (2 * R1 - R2 * (1 - 3 * a2 / sqrt_term) - R3 * (1 + 3 * a2 / sqrt_term)) * 0.5 * self.beam_parameters.gamma_rel**2
        Sx: np.ndarray = (2 * R1 - R2 * (1 + 3 * a2 / sqrt_term) - R3 * (1 - 3 * a2 / sqrt_term)) * 0.5
        Sxp: np.ndarray = 3 * self.beam_parameters.gamma_rel**2 * phix**2 * ax * (R3 - R2) / sqrt_term
        # ----------------------------------------------------------------------------------------------
        # These are the integrands of the integrals in Eq (30-32) in Nagaitsev paper
        Ix_integrand = (
            self.optics.betx
            / (self.optics.circumference * sigx * sigy)
            * (Sx + Sp * (self.optics.dx**2 / self.optics.betx**2 + phix**2) + Sxp)
        )
        Iy_integrand = self.optics.bety / (self.optics.circumference * sigx * sigy) * (R2 + R3 - 2 * R1)
        Ip_integrand = Sp / (self.optics.circumference * sigx * sigy)
        # ----------------------------------------------------------------------------------------------
        # Integrating the integrands above accross the ring to get the desired results
        Ix: float = np.sum(Ix_integrand[:-1] * np.diff(self.optics.s))
        Iy: float = np.sum(Iy_integrand[:-1] * np.diff(self.optics.s))
        Iz: float = np.sum(Ip_integrand[:-1] * np.diff(self.optics.s))
        result = NagaitsevIntegrals(Ix, Iy, Iz)
        # fmt: on
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.elliptic_integrals = result
        return result

    # This is the end of the calculations in 'Nagaitsev_Integrals' from Michail's old code (the last 3 lines essentially)
    # The arguments used to be named Emit_x, Emit_y, Sig_M, BunchL there
    def growth_rates(
        self, geom_epsx: float, geom_epsy: float, sigma_delta: float, bunch_length: float
    ) -> IBSGrowthRates:
        r"""Computes the ``IBS`` growth rates, named :math:`T_x, T_y` and :math:`T_z` in this code base, from Nagaitsev integrals.

        These correspond to the :math:`\dfrac{1}{\tau}` term, for each plane, of Eq (28) in
        :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`, respectively.
        The instance attribute `self.growth_rates` is automatically updated with
        the results of this method.

        .. warning::
            This calculation is done by building on the Nagaitsev integrals. If the
            latter have not been computed yet, this method will raise an error. Please
            remember to call the instance's `integrals` method first.

        .. tip::
            The calculation is done according to the following steps, which are related to different
            equations in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`:

                - Get the Nagaitsev integrals from the instance attributes (integrals of Eq (30-32)).
                - Computes the Coulomb logarithm for the defined beam and optics parameters.
                - Compute the rest of the constant term of Eq (30-32).
                - Compute for each plane the full result of Eq (30-32), respectively.
                - Plug these into Eq (28) and divide by either :math:`\varepsilon_x, \varepsilon_y` or :math:`\sigma_p^{2}` (as relevant) to get :math:`\dfrac{1}{\tau}`.

        Args:
            epsx (float): horizontal geometric emittance in [m].
            epxy (float): vertical geometric emittance in [m].
            sigma_delta (float): momentum spread.
            bunch_length (float): the bunch length in [m].

        Returns:
            An `IBSGrowthRates` object with the computed growth rates for each plane.
        """
        # ----------------------------------------------------------------------------------------------
        # Check that the Nagaitsev integrals have been computed beforehand
        if self.elliptic_integrals is None:
            LOGGER.error("Attempted to compute growth rates without having computed Nagaitsev integrals.")
            raise ValueError(
                "Nagaitsev integrals have not been computed yet, cannot compute growth rates.\n"
                "Please call the `integrals` method first."
            )
        LOGGER.info(
            "Computing IBS growth rates from Nagaitsev integrals for defined beam and optics parameters"
        )
        # ----------------------------------------------------------------------------------------------
        # Check that the Nagaitsev integrals have been computed beforehand
        full_constant_term = self.coulomb_log_full_constant(geom_epsx, geom_epsy, sigma_delta, bunch_length)
        Ix, Iy, Iz = astuple(self.elliptic_integrals)
        Tx: float = Ix * full_constant_term / geom_epsx
        Ty: float = Iy * full_constant_term / geom_epsy
        Tz: float = Iz * full_constant_term / sigma_delta**2
        result = IBSGrowthRates(Tx, Ty, Tz)
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.growth_rates = result
        return result
