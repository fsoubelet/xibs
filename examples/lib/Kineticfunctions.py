"""
This is a very old implementation (much slower) using BJORKEN & MTINGWA (Zenkevich model using BM) model.
"""
import numpy as np
import scipy.integrate as integrate

from scipy.constants import c, e, epsilon_0, hbar, m_p, physical_constants

# TODO: import from xsuite (in PyIBS) for type hints, and add actual type hints


class KineticIBS:
    """A class to encapsulate IBS calculations according to the kinetic formalism."""

    def __init__(self, *args, **kwargs):
        # TODO: some work here, should probably initiate with a twiss or the line
        pass

    def _Phi(self, beta, alpha, eta, eta_d):
        """Go over with Michalis to figure out what this does."""
        return eta_d + alpha * eta / beta

    def _Hi(self, beta, alpha, eta, eta_d):
        """Go over with Michalis to figure out what this does."""
        return (1 / beta) * (eta**2 + (beta * eta_d + alpha * eta) ** 2)

    def set_beam_parameters(self, Ncharges, Npart, Total_Energy) -> None:
        """
        Sets beam parameters in instance from the provided particles object, from xsuite.
        Should be abstracted in a dataclass of its own
        """
        E0p = physical_constants["proton mass energy equivalent in MeV"][0] * 1e-3

        self.Npart = Npart
        self.Ncharg = Ncharges
        self.EnTot = Total_Energy
        self.E_rest = 193.6999
        self.gammar = self.EnTot / self.E_rest
        self.betar = np.sqrt(1 - 1 / self.gammar**2)
        self.mass_i = (self.E_rest * m_p) / E0p  # Ions Mass
        self.c_rad = (self.Ncharg * e) ** 2 / (4 * np.pi * epsilon_0 * c**2 * self.mass_i)

    def set_optic_functions(
        self, position, dels, bet_x, bet_y, alf_x, alf_y, eta_x, eta_dx, eta_y, eta_dy
    ) -> None:
        """
        Sets optics functions in instance from the provided xtrack.TwissTable object.
        Should be abstracted in a dataclass of its own
        """
        self.posit = position
        self.Circu = position[len(position) - 1]
        self.bet_x = bet_x
        self.bet_y = bet_y
        self.eta_x = eta_x
        self.eta_dx = eta_dx
        self.eta_y = eta_y
        self.alf_x = alf_x
        self.alf_y = alf_y
        self.dels = dels
        self.DimT = len(position)
        self.H_x = self._Hi(bet_x, alf_x, eta_x, eta_dx)
        self.H_y = self._Hi(bet_y, alf_y, eta_y, eta_dy)
        self.phi_x = self._Phi(bet_x, alf_x, eta_x, eta_dx)
        self.phi_y = self._Phi(bet_y, alf_y, eta_y, eta_dy)
        self.bx_bar = sum(bet_x * dels) / self.Circu
        self.by_bar = sum(bet_y * dels) / self.Circu
        self.dx_bar = sum(eta_x * dels) / self.Circu
        self.dy_bar = sum(eta_y * dels) / self.Circu
        self.frev = self.betar * c / position[len(position) - 1]

    def meanCoulogConst(self, Emit_x, Emit_y, Sig_M, BunchL):
        """Go over with Michalis to figure out what this does. Calculates mean Coulog constant, which is???"""
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
        Ncon = (
            self.Npart
            * self.c_rad**2
            * c
            / (8 * np.pi * self.betar**3 * self.gammar**4 * Emit_x * Emit_y * BunchL * Sig_M)
        )
        return Ncon * coulog

    def CoulogConst(self, Emit_x, Emit_y, Sig_M, BunchL, ind):
        """Go over with Michalis to figure out what this does. Calculates Coulog constant, which is???"""
        Etrans = 5e8 * (self.gammar * self.EnTot - self.E_rest) * (Emit_x / self.bet_x[ind])
        TempeV = 2.0 * Etrans
        sigxcm = 100 * np.sqrt(Emit_x * self.bet_x[ind] + (self.eta_x[ind] * Sig_M) ** 2)
        sigycm = 100 * np.sqrt(Emit_y * self.bet_y[ind] + (self.eta_y[ind] * Sig_M) ** 2)
        sigtcm = 100 * BunchL
        volume = 8.0 * np.sqrt(np.pi**3) * sigxcm * sigycm * sigtcm
        densty = self.Npart / volume
        debyul = 743.4 * np.sqrt(TempeV / densty) / self.Ncharg
        rmincl = 1.44e-7 * self.Ncharg**2 / TempeV
        rminqm = hbar * c * 1e5 / (2.0 * np.sqrt(2e-3 * Etrans * self.E_rest))
        rmin = max(rmincl, rminqm)
        rmax = min(sigxcm, debyul)
        coulog = np.log(rmax / rmin)
        Ncon = (
            self.Npart
            * self.c_rad**2
            * c
            / (8 * np.pi * self.betar**3 * self.gammar**4 * Emit_x * Emit_y * BunchL * Sig_M)
        )
        return Ncon * coulog

    def L_matrix(self, ind, Emit_x, Emit_y, Sig_M, lam_lim=20.0) -> None:
        """Go over with Michalis to figure out what this does. Calculates an L matrix constant (which is???) and sets it into the instance itself."""
        T1 = self.bet_x[ind] / Emit_x
        T2 = self.bet_y[ind] / Emit_y
        L = np.array(
            [
                [T1, 0.0, -T1 * self.phi_x[ind] * self.gammar],
                [0.0, T2, -T2 * self.phi_y[ind] * self.gammar],
                [
                    -T1 * self.phi_x[ind] * self.gammar,
                    -T2 * self.phi_y[ind] * self.gammar,
                    self.gammar**2 * (self.H_x[ind] / Emit_x + self.H_y[ind] / Emit_y + 1 / Sig_M**2),
                ],
            ]
        )
        self.LL = L
        self.uplim = np.amax(self.LL) * lam_lim

    def det_L(self, lam):
        """Go over with Michalis to figure out what this does."""
        det = (
            ((self.LL[0][0] + lam) * (self.LL[1][1] + lam) * (self.LL[2][2] + lam))
            - (self.LL[0][2] * (self.LL[1][1] + lam) * self.LL[2][0])
            - ((self.LL[0][0] + lam) * self.LL[1][2] * self.LL[2][1])
        )
        return det

    def fr_x(self, lam):
        """Go over with Michalis to figure out what this does."""
        det = self.det_L(lam)
        return (
            np.sqrt(lam / det)
            / det
            * ((self.LL[1][1] + lam) * (self.LL[2][2] + lam) - self.LL[1][2] * self.LL[1][2])
        )

    def fr_y(self, lam):
        """Go over with Michalis to figure out what this does."""
        det = self.det_L(lam)
        return (
            np.sqrt(lam / det)
            / det
            * ((self.LL[0][0] + lam) * (self.LL[2][2] + lam) - self.LL[0][2] * self.LL[0][2])
        )

    def fr_z(self, lam):
        """Go over with Michalis to figure out what this does."""
        det = self.det_L(lam)
        return np.sqrt(lam / det) / det * ((self.LL[0][0] + lam) * (self.LL[1][1] + lam))

    def fr_xz(self, lam):
        """Go over with Michalis to figure out what this does."""
        det = self.det_L(lam)
        return np.sqrt(lam / det) / det * ((self.LL[1][1] + lam) * self.LL[0][2] * (-1.0))

    def friction(self):
        """Go over with Michalis to figure out what this does."""
        kx = integrate.quad(self.fr_x, 0, self.uplim)[0]
        ky = integrate.quad(self.fr_y, 0, self.uplim)[0]
        kz = integrate.quad(self.fr_z, 0, self.uplim)[0]
        kxz = integrate.quad(self.fr_xz, 0, self.uplim)[0]
        return kx, ky, kz, kxz

    def diff_xx(self, lam):
        """Go over with Michalis to figure out what this does."""
        det = self.det_L(lam)
        return (
            np.sqrt(lam / det)
            / det
            * (
                (self.LL[0][0] + lam) * (self.LL[2][2] + lam)
                - self.LL[0][2] * self.LL[0][2]
                + (self.LL[0][0] + lam) * (self.LL[1][1] + lam)
            )
        )

    def diff_yy(self, lam):
        """Go over with Michalis to figure out what this does."""
        det = self.det_L(lam)
        return (
            np.sqrt(lam / det)
            / det
            * (
                (self.LL[1][1] + lam) * (self.LL[2][2] + lam)
                - self.LL[1][2] * self.LL[1][2]
                + (self.LL[0][0] + lam) * (self.LL[1][1] + lam)
            )
        )

    def diff_zz(self, lam):
        """Go over with Michalis to figure out what this does."""
        det = self.det_L(lam)
        return (
            np.sqrt(lam / det)
            / det
            * (
                (self.LL[1][1] + lam) * (self.LL[2][2] + lam)
                - self.LL[1][2] * self.LL[1][2]
                + (self.LL[0][0] + lam) * (self.LL[2][2] + lam)
            )
        )

    def diff_zx(self, lam):
        """Go over with Michalis to figure out what this does."""
        det = self.det_L(lam)
        return np.sqrt(lam / det) / det * (self.LL[1][1] + lam) * self.LL[0][2]

    def diff_zy(self, lam):
        """Go over with Michalis to figure out what this does."""
        det = self.det_L(lam)
        return np.sqrt(lam / det) / det * (self.LL[0][0] + lam) * self.LL[1][2]

    def diffusion(self):
        """Go over with Michalis to figure out what this does."""
        dxx = integrate.quad(self.diff_xx, 0, self.uplim)[0]
        dyy = integrate.quad(self.diff_yy, 0, self.uplim)[0]
        dzz = integrate.quad(self.diff_zz, 0, self.uplim)[0]
        dzx = integrate.quad(self.diff_zx, 0, self.uplim)[0]
        dzy = integrate.quad(self.diff_zy, 0, self.uplim)[0]
        return dxx, dyy, dzz, dzx, dzy

    def evaluate_coefficients(self, Emit_x, Emit_y, Sig_M, BunchL) -> None:
        """
        Go over with Michalis to figure out what this does.
        Supposedly calculates kinetic coefficients and sets them into the instance itself.
        """
        GRx, GRy, GRz = [], [], []
        FRx, FRy, FRz = [], [], []
        for j in range(self.DimT):
            ACon = self.CoulogConst(Emit_x, Emit_y, Sig_M, BunchL, j)
            # ACon = self.meanCoulogConst(Emit_x, Emit_y, Sig_M, BunchL)

            self.L_matrix(j, Emit_x, Emit_y, Sig_M, 20.0)

            # !---------- Friction ----------!
            Fx, Fy, Fz, Fxz = self.friction()
            Kxx = ACon * Fx * self.bet_x[j] / Emit_x
            Kyy = ACon * Fy * self.bet_y[j] / Emit_y
            Kzz = ACon * Fz / Sig_M**2
            Kxz = ACon * Fxz

            # !--------- Diffusion ---------!
            dxx, dyy, dzz, dzx, dzy = self.diffusion()
            Dxx = ACon * dxx
            Dyy = ACon * dyy
            Dzz = ACon * dzz
            Dxz = ACon * dzx
            Dyz = ACon * dzy

            GRx.append(
                self.bet_x[j] / Emit_x * Dxx
                + self.gammar**2 * self.H_x[j] / Emit_x * Dzz
                - 2 * self.bet_x[j] * self.phi_x[j] * self.gammar / Emit_x * Dxz
            )
            GRy.append(
                self.bet_y[j] / Emit_y * Dyy
                + self.gammar**2 * self.H_y[j] / Emit_y * Dzz
                - 2 * self.bet_y[j] * self.phi_y[j] * self.gammar / Emit_y * Dyz
            )
            GRz.append(self.gammar**2 / Sig_M**2 * Dzz)

            FRx.append(
                self.bet_x[j] / Emit_x * (2 * Emit_x / self.bet_x[j] * Kxx)
                + self.gammar**2 * self.H_x[j] / Emit_x * (2 * Sig_M**2 * Kzz)
                - 2 * self.bet_x[j] * self.phi_x[j] * self.gammar / Emit_x * 2 * Kxz
            )
            FRy.append(
                self.bet_y[j] / Emit_y * (2 * Emit_y / self.bet_y[j] * Kyy)
                + self.gammar**2 * self.H_y[j] / Emit_y * (2 * Sig_M**2 * Kzz)
            )
            FRz.append(2 * self.gammar**2 * Kzz)

        # !--------- Diffusion ---------!
        self.Dx = np.sum(GRx * self.dels) / self.Circu
        self.Dy = np.sum(GRy * self.dels) / self.Circu
        self.Dz = np.sum(GRz * self.dels) / self.Circu

        # !---------- Friction ----------!
        self.Fx = np.sum(FRx * self.dels) / self.Circu
        self.Fy = np.sum(FRy * self.dels) / self.Circu
        self.Fz = np.sum(FRz * self.dels) / self.Circu

    def kinetic_kick(self, part_coord_dict, alpha_x, alpha_y, dt) -> None:
        """Applies the kinetic kick to the particles objects?? Unclear how the dict fits into xsuite structures yet."""
        # print(part_coord_dict.shape[0])
        # print(p_std_x)
        # print(self.Dx)

        Ran1 = np.random.normal(loc=0, scale=1, size=part_coord_dict.shape[0])
        Ran2 = np.random.normal(loc=0, scale=1, size=part_coord_dict.shape[0])
        Ran3 = np.random.normal(loc=0, scale=1, size=part_coord_dict.shape[0])

        p_std_x = np.std(part_coord_dict["px"]) / np.sqrt(1 + alpha_x**2)
        p_std_y = np.std(part_coord_dict["py"]) / np.sqrt(1 + alpha_y**2)
        p_std_z = np.std(part_coord_dict["pz"])

        # !---------- Friction ----------!
        part_coord_dict["px"] -= self.Fx * part_coord_dict["px"] * dt
        part_coord_dict["py"] -= self.Fy * part_coord_dict["py"] * dt
        part_coord_dict["pz"] -= self.Fz * part_coord_dict["pz"] * dt

        # !---------- Diffusion ----------!
        part_coord_dict["px"] += p_std_x * np.sqrt(2 * dt * self.Dx) * Ran1
        part_coord_dict["py"] += p_std_y * np.sqrt(2 * dt * self.Dy) * Ran2
        part_coord_dict["pz"] += p_std_z * np.sqrt(2 * dt * self.Dz) * Ran3
