"""
This is the old implementation of Michalis, kept here as a private module to benchmark against in tests.
"""
import warnings

from typing import Tuple

import numpy as np
import scipy
import scipy.integrate as integrate

from scipy.constants import c, hbar, physical_constants
from scipy.interpolate import interp1d

warnings.warn(
    "You should not be using this module unless you are a developper "
    "of this package and know what you are doing."
)


class MichalisIBS:
    """
    .. versionadded:: 0.2.0

    A class to encapsulate IBS calculations according to the Nagaitsev formalism.
    """

    def __init__(self, *args, **kwargs):
        pass

    def _Phi(self, beta, alpha, eta, eta_d):
        """Go over with Michalis to figure out what this does."""
        return eta_d + alpha * eta / beta

    def set_beam_parameters(self, particles) -> None:
        """
        Sets beam parameters in instance from the provided particles object, from xsuite.
        Should be abstracted in a dataclass of its own
        """
        self.Npart = particles.weight[0] * particles.gamma0.shape[0]
        self.Ncharg = particles.q0
        self.E_rest = particles.mass0 * 1e-9
        self.EnTot = np.sqrt(particles.p0c[0] ** 2 + particles.mass0**2) * 1e-9
        self.gammar = particles.gamma0[0]
        self.betar = particles.beta0[0]
        # self.c_rad  = physical_constants["classical electron radius"][0]
        E0p = physical_constants["proton mass energy equivalent in MeV"][0] * 1e-3
        particle_mass_GEV = particles.mass0 * 1e-9
        mi = (particle_mass_GEV * scipy.constants.m_p) / E0p
        # classical radius, can get from xpart.Particles now
        self.c_rad = (particles.q0 * scipy.constants.e) ** 2 / (
            4 * np.pi * scipy.constants.epsilon_0 * scipy.constants.c**2 * mi
        )

    def set_optic_functions(self, twiss) -> None:
        """
        Sets optics functions in instance from the provided xtrack.TwissTable object.
        Should be abstracted in a dataclass of its own
        """
        self.posit = twiss["s"]
        self.Circu = twiss["s"][-1]
        self.bet_x = twiss["betx"]
        self.bet_y = twiss["bety"]
        self.alf_x = twiss["alfx"]
        self.alf_y = twiss["alfy"]
        self.eta_x = twiss["dx"]  # eta_x because notations from old papers
        self.eta_dx = twiss["dpx"]
        self.eta_y = twiss["dy"]
        self.eta_dy = twiss["dpy"]
        self.slip = twiss["slip_factor"]
        self.phi_x = self._Phi(twiss["betx"], twiss["alfx"], twiss["dx"], twiss["dpx"])
        self.frev = self.betar * c / self.Circu
        # Interpolated functions for the calculation below
        bx_b = interp1d(twiss["s"], twiss["betx"])
        by_b = interp1d(twiss["s"], twiss["bety"])
        dx_b = interp1d(twiss["s"], twiss["dx"])
        dy_b = interp1d(twiss["s"], twiss["dy"])
        # Below is the average beta and dispersion functions - better here than a simple np.mean calculation because the latter doesn't take in consideration element lengths etc
        # These are ONLY USED in the CoulogConst function
        self.bx_bar = integrate.quad(bx_b, twiss["s"][0], twiss["s"][-1])[0] / self.Circu
        self.by_bar = integrate.quad(by_b, twiss["s"][0], twiss["s"][-1])[0] / self.Circu
        self.dx_bar = integrate.quad(dx_b, twiss["s"][0], twiss["s"][-1])[0] / self.Circu
        self.dy_bar = integrate.quad(dy_b, twiss["s"][0], twiss["s"][-1])[0] / self.Circu

    def CoulogConst(self, Emit_x, Emit_y, Sig_M, BunchL):
        """
        This is the full constant factor (building on Coulomb Log (constant) from Eq 9 in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`).
        Calculates Coulog constant, then log and returns multiplied by ???"""
        Etrans = 5e8 * (self.gammar * self.EnTot - self.E_rest) * (Emit_x / self.bx_bar)
        TempeV = 2.0 * Etrans
        sigxcm = 100 * np.sqrt(Emit_x * self.bx_bar + (self.dx_bar * Sig_M) ** 2)
        sigycm = 100 * np.sqrt(Emit_y * self.by_bar + (self.dy_bar * Sig_M) ** 2)
        sigtcm = 100 * BunchL
        volume = 8.0 * np.sqrt(np.pi**3) * sigxcm * sigycm * sigtcm
        densty = self.Npart / volume
        debyul = 743.4 * np.sqrt(TempeV / densty) / self.Ncharg
        rmincl = 1.44e-7 * self.Ncharg**2 / TempeV
        rminqm = hbar * c * 1e5 / (2.0 * np.sqrt(2e-3 * Etrans * self.E_rest))
        rmin = max(rmincl, rminqm)
        rmax = min(sigxcm, debyul)
        coulog = np.log(rmax / rmin)
        Ncon = self.Npart * self.c_rad**2 * c / (12 * np.pi * self.betar**3 * self.gammar**5 * BunchL)
        return Ncon * coulog

    def RDiter(self, x, y, z):
        """
        Elliptic integral calculation with iterative method.
        This is the R_D calculation from Eq (4) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.
        This looks like it can be much more optimized. This is an implementation that was found by Michalis (in Cpp and adapted).
        Found in ref [5] of Nagaitsev paper (sues ref [4]), they give an iterative procedure to calculate Eq (4).
        Some powerpoints from Michalis in ABP group meeting mention how this is calculated.

        Args:
            x (float): the Lambda1 value in Nagaitsev paper. Eigen values of the A matrix in Eq (2) which comes from B&M. In B&M it is L matrix.
            y (float): the Lambda2 value in Nagaitsev paper. Eigen values of the A matrix in Eq (2) which comes from B&M. In B&M it is L matrix.
            z (float): the Lambda3 value in Nagaitsev paper. Eigen values of the A matrix in Eq (2) which comes from B&M. In B&M it is L matrix.

        This is because Nagaitsev shows we can calculate the R_D integral at 3 different specific points and have the whole.
        """
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

    # Run if you want the IBS growth rates
    def Nagaitsev_Integrals(self, Emit_x, Emit_y, Sig_M, BunchL) -> Tuple[float, float, float]:
        """Computes the Nagaitsev integrals Ix, Iy and Ip, and then the IBS growth rates.

        -> Calculates various constants from Eq (18-21) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`
        -> Calculates the R_D terms with `RDIter` function
        -> Plugs it into Eq (33-25) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`
        -> Computes the Nagaitsev integral terms - Eq (30-32)
        -> Computes the IBS growth rates - Eq (28) (integrate the terms, multiply by Coulomb log and divide by emittance)
        """
        # Constants from Eq (18-21)
        const = self.CoulogConst(Emit_x, Emit_y, Sig_M, BunchL)  # this is a float
        # For each of the following (until denom), they are an np.ndarray with one value per element in the lattice
        sigx = np.sqrt(self.bet_x * Emit_x + (self.eta_x * Sig_M) ** 2)
        sigy = np.sqrt(self.bet_y * Emit_y + (self.eta_y * Sig_M) ** 2)
        ax = self.bet_x / Emit_x
        ay = self.bet_y / Emit_y
        a_s = ax * (self.eta_x**2 / self.bet_x**2 + self.phi_x**2) + 1 / Sig_M**2
        a1 = (ax + self.gammar**2 * a_s) / 2.0
        a2 = (ax - self.gammar**2 * a_s) / 2.0
        denom = np.sqrt(a2**2 + self.gammar**2 * ax**2 * self.phi_x**2)
        # --------------------------------------------------------------------------------
        # This is from Eq (22-24) in Nagaitsev paper, eigen values of A matrix (L matrix in B&M)
        # Similarly these are each an np.ndarray with one value per element in the lattice
        l1 = ay
        l2 = a1 + denom
        l3 = a1 - denom
        # --------------------------------------------------------------------------------
        # This is from Eq (25-27) in Nagaitsev paper
        # Once again these are each an np.ndarray with one value per element in the lattice
        R1 = self.RDiter(1 / l2, 1 / l3, 1 / l1) / l1
        R2 = self.RDiter(1 / l3, 1 / l1, 1 / l2) / l2
        R3 = 3 * np.sqrt(l1 * l2 / l3) - l1 * R1 / l3 - l2 * R2 / l3
        # --------------------------------------------------------------------------------
        # This is Eq (33-35) in Nagaitsev paper - (partial?) growth rates
        # These are each an np.ndarray with one value per element in the lattice
        Nagai_Sp = (2 * R1 - R2 * (1 - 3 * a2 / denom) - R3 * (1 + 3 * a2 / denom)) * 0.5 * self.gammar**2
        Nagai_Sx = (2 * R1 - R2 * (1 + 3 * a2 / denom) - R3 * (1 - 3 * a2 / denom)) * 0.5
        Nagai_Sxp = 3 * self.gammar**2 * self.phi_x**2 * ax * (R3 - R2) / denom
        # --------------------------------------------------------------------------------
        # THIS IS THE INTEGRALS, USED BELOW TO CALCULATE THE GROWTH RATES
        # Actually these are still the integrands, the integration is done at the next step
        # This is Eq (30-32) then directly plugged into Eq (28) in Nagaitsev paper
        Ixi = (
            self.bet_x
            / (self.Circu * sigx * sigy)
            * (Nagai_Sx + Nagai_Sp * (self.eta_x**2 / self.bet_x**2 + self.phi_x**2) + Nagai_Sxp)
        )
        Iyi = self.bet_y / (self.Circu * sigx * sigy) * (R2 + R3 - 2 * R1)
        Ipi = Nagai_Sp / (self.Circu * sigx * sigy)
        # --------------------------------------------------------------------------------
        # This is were we plug the last part in Eq (28) -> division by the emittance
        # technically the first part of the calculation are the Nagaitsev Integrals
        # The growth rates are Ix, Iy, Ip computed with the constant factor and the emittances
        Ix: float = np.sum(Ixi[:-1] * np.diff(self.posit)) * const / Emit_x
        Iy: float = np.sum(Iyi[:-1] * np.diff(self.posit)) * const / Emit_y
        Ip: float = np.sum(Ipi[:-1] * np.diff(self.posit)) * const / Sig_M**2
        # figure out with Michalis why the integration is commented out
        # This is to save time, as it is so much move intensive
        # Ix = integrate.simps(Ixi, self.posit) * const / Emit_x
        # Iy = integrate.simps(Iyi, self.posit) * const / Emit_y
        # Ip = integrate.simps(Ipi, self.posit) * const / Sig_M**2
        # THIS IS THE GROWTH RATES!!!!!!
        return Ix, Iy, Ip

    # Run to calculate and save the growth rates; used for the emittance evolution
    def calculate_integrals(self, Emit_x, Emit_y, Sig_M, BunchL) -> None:
        """Computes the Nagaitsev GROWTH RATES HERE Ixx, Iyy and Ipp, and stores them in the instance itself."""
        # Remember: this is growth rates!
        self.Ixx, self.Iyy, self.Ipp = self.Nagaitsev_Integrals(Emit_x, Emit_y, Sig_M, BunchL)

    # Run if you want to evaluate the emittance evolution using Nagaitsev's Integrals.
    def emit_evol(self, Emit_x, Emit_y, Sig_M, BunchL, dt) -> Tuple[float, float, float]:
        """Computes the emittance evolutions in 3D from the Nagaitsev integrals and resulting growth rates."""
        Evolemx = Emit_x * np.exp(dt * float(self.Ixx))
        Evolemy = Emit_y * np.exp(dt * float(self.Iyy))
        EvolsiM = Sig_M * np.exp(dt * float(0.5 * self.Ipp))
        return Evolemx, Evolemy, EvolsiM

    # Run if you want to evaluate the emittance evolution using Nagaitsev's Integrals, including Synchrotron Radiation.
    # Give damping times in [s], not turns!!!
    def emit_evol_with_SR(
        self, Emit_x, Emit_y, Sig_M, BunchL, EQemitX, EQemitY, EQsigmM, tau_x, tau_y, tau_s, dt
    ) -> Tuple[float, float, float]:
        """Computes the emittance evolutions from the Nagaitsev integrals, including the effects of synchrotron radiation."""
        Evolemx = (
            -EQemitX
            + np.exp(dt * 2 * (float(self.Ixx / 2.0) - 1.0 / tau_x))
            * (EQemitX + Emit_x * (float(self.Ixx / 2.0) * tau_x - 1.0))
        ) / (float(self.Ixx / 2.0) * tau_x - 1.0)
        Evolemy = (
            -EQemitY
            + np.exp(dt * 2 * (float(self.Iyy / 2.0) - 1.0 / tau_y))
            * (EQemitY + Emit_y * (float(self.Iyy / 2.0) * tau_y - 1.0))
        ) / (float(self.Iyy / 2.0) * tau_y - 1.0)
        EvolsiM = np.sqrt(
            (
                -(EQsigmM**2)
                + np.exp(dt * 2 * (float(self.Ipp / 2.0) - 1.0 / tau_s))
                * (EQsigmM**2 + Sig_M**2 * (float(self.Ipp / 2.0) * tau_s - 1.0))
            )
            / (float(self.Ipp / 2.0) * tau_s - 1.0)
        )

        return Evolemx, Evolemy, EvolsiM

    def line_density(self, n_slices, particles):
        """Calculates line density, implementation from Michalis and Hannes. Idea came from Eq 8 in https://journals.aps.org/prab/abstract/10.1103/PhysRevSTAB.13.091001 (CITE)
        -> Get particles coordinates
        -> Determine binning of coordinates for histogram (getting bin edges and centers)
        -> Calculate the rms bunch length as the standard deviation of the distribution
        -> Does an interpolation of the histogram with wanted number of bins and returns an interpolated function to apply kicks later on.
        Weigths are applied heavier at the center of the interpolated array so that for kicks particles at the center of the distribution are more affected.
        """
        zeta = particles.zeta[particles.state > 0]
        z_cut_head = np.max(zeta)
        z_cut_tail = np.min(zeta)
        slice_width = (z_cut_head - z_cut_tail) / float(n_slices)

        bin_edges = np.linspace(
            z_cut_tail - 1e-7 * slice_width,
            z_cut_head + 1e-7 * slice_width,
            num=n_slices + 1,
            dtype=np.float64,
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        bunch_length_rms = np.std(zeta)
        factor_distribution = bunch_length_rms * 2 * np.sqrt(np.pi)

        counts_normed, bin_edges = np.histogram(zeta, bin_edges, density=True)
        Rho_normed = np.interp(zeta, bin_centers, counts_normed * factor_distribution)
        # kick_factor_normed = np.mean(Rho_normed)

        return Rho_normed

    # ! ~~~~~~~~~~~~~~~~ Simple Kicks ~~~~~~~~~~~~~~~~~~ !
    def emit_evol_simple_kicks(self, particles) -> Tuple[float, float, float]:
        """Computes the simple kick evolutions for a particles object from xsuite."""
        Sig_x = np.std(particles.x[particles.state > 0])
        Sig_y = np.std(particles.y[particles.state > 0])
        Sig_zeta = np.std(particles.zeta[particles.state > 0])
        Sig_delta = np.std(particles.delta[particles.state > 0])

        Emit_x = (Sig_x**2 - (self.eta_x[0] * Sig_delta) ** 2) / self.bet_x[0]
        Emit_y = Sig_y**2 / self.bet_y[0]

        Sig_px_norm = np.std(particles.px[particles.state > 0]) / np.sqrt(1 + self.alf_x[0] ** 2)
        Sig_py_norm = np.std(particles.py[particles.state > 0]) / np.sqrt(1 + self.alf_y[0] ** 2)

        Ixx, Iyy, Ipp = self.Nagaitsev_Integrals(Emit_x, Emit_y, Sig_delta, Sig_zeta)

        if Ixx < 0:
            Ixx = 0
        if Iyy < 0:
            Iyy = 0
        if Ipp < 0:
            Ipp = 0

        DSx = Sig_px_norm * np.sqrt(2 * Ixx / self.frev)
        DSy = Sig_py_norm * np.sqrt(2 * Iyy / self.frev)
        DSz = Sig_delta * np.sqrt(2 * Ipp / self.frev) * self.betar**2

        return DSx, DSy, DSz

    # Run to calculate and save the simple kick strengths; to be used for the simple kick
    def calculate_simple_kick(self, particles) -> None:
        """Computes the simple kick evolutions from Nagaitsev integrals (via method above) and stores them in the instance itself."""
        self.DSx, self.DSy, self.DSz = self.emit_evol_simple_kicks(particles)

    # Run !EVERY TURN! to apply the simple kick. Needs adjustment if it is not every turn
    def apply_simple_kick(self, particles) -> None:
        """Applies the computed simple kick evolutions from Nagaitsev integrals (via method above) to the particle objects."""
        rho = self.line_density(40, particles)
        Dkick_x = np.random.normal(
            loc=0, scale=self.DSx, size=particles.px[particles.state > 0].shape[0]
        ) * np.sqrt(rho)
        Dkick_y = np.random.normal(
            loc=0, scale=self.DSy, size=particles.py[particles.state > 0].shape[0]
        ) * np.sqrt(rho)
        Dkick_p = np.random.normal(
            loc=0, scale=self.DSz, size=particles.delta[particles.state > 0].shape[0]
        ) * np.sqrt(rho)

        particles.px[particles.state > 0] += Dkick_x
        particles.py[particles.state > 0] += Dkick_y
        particles.delta[particles.state > 0] += Dkick_p

    # ! ~~~~~~~~~~~~~~~~ Kinetic Kicks ~~~~~~~~~~~~~~~~~~ !
    def Kinetic_Coefficients(
        self, Emit_x, Emit_y, Sig_M, BunchL
    ) -> Tuple[float, float, float, float, float, float]:
        """Computes the kinetic coefficients based on emittances."""
        const = self.CoulogConst(Emit_x, Emit_y, Sig_M, BunchL)
        sigx = np.sqrt(self.bet_x * Emit_x + (self.eta_x * Sig_M) ** 2)
        sigy = np.sqrt(self.bet_y * Emit_y + (self.eta_y * Sig_M) ** 2)
        ax = self.bet_x / Emit_x
        ay = self.bet_y / Emit_y
        a_s = ax * (self.eta_x**2 / self.bet_x**2 + self.phi_x**2) + 1 / Sig_M**2
        a1 = (ax + self.gammar**2 * a_s) / 2.0
        a2 = (ax - self.gammar**2 * a_s) / 2.0
        denom = np.sqrt(a2**2 + self.gammar**2 * ax**2 * self.phi_x**2)
        # --------------------------------------------------------------------------------
        l1 = ay
        l2 = a1 + denom
        l3 = a1 - denom
        # --------------------------------------------------------------------------------
        R1 = self.RDiter(1 / l2, 1 / l3, 1 / l1) / l1
        R2 = self.RDiter(1 / l3, 1 / l1, 1 / l2) / l2
        R3 = 3 * np.sqrt(l1 * l2 / l3) - l1 * R1 / l3 - l2 * R2 / l3
        # --------------------------------------------------------------------------------
        D_Sp = 0.5 * self.gammar**2 * (2 * R1 + R2 * (1 + a2 / denom) + R3 * (1 - a2 / denom))
        F_Sp = 1.0 * self.gammar**2 * (R2 * (1 - a2 / denom) + R3 * (1 + a2 / denom))
        D_Sx = 0.5 * (2 * R1 + R2 * (1 - a2 / denom) + R3 * (1 + a2 / denom))
        F_Sx = 1.0 * (R2 * (1 + a2 / denom) + R3 * (1 - a2 / denom))
        D_Sxp = 3.0 * self.gammar**2 * self.phi_x**2 * ax * (R3 - R2) / denom

        Dxi = (
            self.bet_x
            / (self.Circu * sigx * sigy)
            * (D_Sx + D_Sp * (self.eta_x**2 / self.bet_x**2 + self.phi_x**2) + D_Sxp)
        )
        Fxi = (
            self.bet_x
            / (self.Circu * sigx * sigy)
            * (F_Sx + F_Sp * (self.eta_x**2 / self.bet_x**2 + self.phi_x**2))
        )
        Dyi = self.bet_y / (self.Circu * sigx * sigy) * (R2 + R3)
        Fyi = self.bet_y / (self.Circu * sigx * sigy) * (2 * R1)
        Dzi = D_Sp / (self.Circu * sigx * sigy)
        Fzi = F_Sp / (self.Circu * sigx * sigy)

        # Dx = np.sum(Dxi * self.dels) * const / Emit_x
        # Fx = np.sum(Fxi * self.dels) * const / Emit_x
        # Dy = np.sum(Dyi * self.dels) * const / Emit_y
        # Fy = np.sum(Fyi * self.dels) * const / Emit_y
        # Dz = np.sum(Dzi * self.dels) * const / Sig_M**2 #* 2. for coasting
        # Fz = np.sum(Fzi * self.dels) * const / Sig_M**2 #* 2. for coasting
        # why no integration calculation here either
        Dx = np.sum(Dxi[:-1] * np.diff(self.posit)) * const / Emit_x
        Dy = np.sum(Dyi[:-1] * np.diff(self.posit)) * const / Emit_y
        Dz = np.sum(Dzi[:-1] * np.diff(self.posit)) * const / Sig_M**2
        Fx = np.sum(Fxi[:-1] * np.diff(self.posit)) * const / Emit_x
        Fy = np.sum(Fyi[:-1] * np.diff(self.posit)) * const / Emit_y
        Fz = np.sum(Fzi[:-1] * np.diff(self.posit)) * const / Sig_M**2
        # Dx = integrate.simps(Dxi, self.posit) * const / Emit_x
        # Dy = integrate.simps(Dyi, self.posit) * const / Emit_y
        # Dz = integrate.simps(Dzi, self.posit) * const / Sig_M**2
        # Fx = integrate.simps(Fxi, self.posit) * const / Emit_x
        # Fy = integrate.simps(Fyi, self.posit) * const / Emit_y #* 2. for coasting
        # Fz = integrate.simps(Fzi, self.posit) * const / Sig_M**2 #* 2. for coasting

        self.kinTx, self.kinTy, self.kinTz = Dx - Fx, Dy - Fy, Dz - Fz
        return Dx, Fx, Dy, Fy, Dz, Fz  # units [1/s]

    # Run to calculate and save the kinetic coefficients; to be used for the kinetic kick
    def calculate_kinetic_coefficients(self, particles) -> None:
        """Computes the kinetic coefficients based on emittances and stores them in the instance itself."""
        Sig_x = np.std(particles.x[particles.state > 0])
        Sig_y = np.std(particles.y[particles.state > 0])
        Sig_zeta = np.std(particles.zeta[particles.state > 0])
        Sig_delta = np.std(particles.delta[particles.state > 0])

        Emit_x = (Sig_x**2 - (self.eta_x[0] * Sig_delta) ** 2) / self.bet_x[0]
        Emit_y = Sig_y**2 / self.bet_y[0]

        self.Dx, self.Fx, self.Dy, self.Fy, self.Dz, self.Fz = self.Kinetic_Coefficients(
            Emit_x, Emit_y, Sig_delta, Sig_zeta
        )

    # Run to apply the kinetic kick.
    def apply_kinetic_kick(self, particles) -> None:
        """Applies the kinetic coefficients based on emittances (via method above) to the particle objects."""
        dt = 1 / self.frev  # needs to be changed.
        Ran1 = np.random.normal(loc=0, scale=1, size=particles.px[particles.state > 0].shape[0])
        Ran2 = np.random.normal(loc=0, scale=1, size=particles.py[particles.state > 0].shape[0])
        Ran3 = np.random.normal(loc=0, scale=1, size=particles.delta[particles.state > 0].shape[0])

        Sig_px_norm = np.std(particles.px[particles.state > 0]) / np.sqrt(1 + self.alf_x[0] ** 2)
        Sig_py_norm = np.std(particles.py[particles.state > 0]) / np.sqrt(1 + self.alf_y[0] ** 2)
        Sig_delta = np.std(particles.delta[particles.state > 0])

        rho = self.line_density(40, particles)  # number of slices

        # !---------- Friction ----------!
        particles.px[particles.state > 0] -= (
            self.Fx
            * (particles.px[particles.state > 0] - np.mean(particles.px[particles.state > 0]))
            * dt
            * rho
        )  # kick units [1]
        particles.py[particles.state > 0] -= (
            self.Fy
            * (particles.py[particles.state > 0] - np.mean(particles.py[particles.state > 0]))
            * dt
            * rho
        )  # kick units [1]
        particles.delta[particles.state > 0] -= (
            self.Fz
            * (particles.delta[particles.state > 0] - np.mean(particles.delta[particles.state > 0]))
            * dt
            * rho
        )  # kick units [1]

        # !---------- Diffusion ----------!
        particles.px[particles.state > 0] += (
            Sig_px_norm * np.sqrt(2 * dt * self.Dx) * Ran1 * np.sqrt(rho)
        )  # kick units [1]
        particles.py[particles.state > 0] += (
            Sig_py_norm * np.sqrt(2 * dt * self.Dy) * Ran2 * np.sqrt(rho)
        )  # kick units [1]
        particles.delta[particles.state > 0] += (
            Sig_delta * np.sqrt(2 * dt * self.Dz) * Ran3 * np.sqrt(rho)
        )  # kick units [1]
