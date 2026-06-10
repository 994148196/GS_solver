"""
gspack.profiles
===============
Plasma current profiles — upgraded with:
  • ConstrainPaxisIp  (original — fix p₀ and Ip)
  • ConstrainBetapIp  (new — fix βp and Ip)
  • ConstrainRotation (new — fix p₀, Ip, and toroidal rotation Ω(ψ))
  • ProfilesPprimeFfprime (new — specify p'(ψ) and ff'(ψ) directly)

All profiles share the same Jtor canonical form:
  Jtor = L [β₀ R/R_axis + (1-β₀) R_axis/R] j̃(ψN)
  j̃(ψN) = (1 - ψN^αm)^αn
"""

import numpy as np
from scipy.integrate import quad, romb
from .backend import MU0
from . import critical


class _ProfileBase:
    """Common infrastructure for all profile classes."""

    alpha_m : float = 1.0
    alpha_n : float = 2.0
    Raxis   : float = 1.0

    # Set by Jtor(); used by pressure/fpol
    L         : float = 0.0
    Beta0     : float = 0.0
    psi_axis  : float = 0.0
    psi_bndry : float = 0.0

    def _shape(self, psiN):
        return (1.0 - np.clip(psiN, 0.0, 1.0)**self.alpha_m)**self.alpha_n

    def pprime(self, psiN):
        return self.L * self.Beta0 / self.Raxis * self._shape(psiN)

    def ffprime(self, psiN):
        return MU0 * self.L * (1.0 - self.Beta0) * self.Raxis * self._shape(psiN)

    def fvac(self):
        return self._fvac

    def pressure(self, psiN):
        """p(ψN) by integrating p'."""
        psiN = np.asarray(psiN, dtype=float)
        scalar = psiN.ndim == 0
        psiN = np.atleast_1d(psiN)
        dpsi = self.psi_bndry - self.psi_axis
        result = np.array([
            quad(self.pprime, float(p), 1.0)[0] * dpsi
            for p in psiN.ravel()
        ]).reshape(psiN.shape)
        return float(result[0]) if scalar else result

    def fpol(self, psiN):
        """f(ψN) = R·Bφ by integrating ff'."""
        psiN = np.asarray(psiN, dtype=float)
        scalar = psiN.ndim == 0
        psiN = np.atleast_1d(psiN)
        dpsi = self.psi_bndry - self.psi_axis
        f2vac = self.fvac()**2
        result = np.array([
            np.sqrt(max(2.0 * quad(self.ffprime, float(p), 1.0)[0] * dpsi + f2vac, 0.0))
            for p in psiN.ravel()
        ]).reshape(psiN.shape)
        return float(result[0]) if scalar else result

    def _solve_constraints(self, jtorshape, R, dR, dZ, dpsi):
        """
        Shared solver for (L, Beta0) from plasma current constraints.
        Returns (L, Beta0, LBeta0).
        """
        # Overridden in subclasses that need betap constraint
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
#  ConstrainPaxisIp  — fix pressure on axis + Ip
# ─────────────────────────────────────────────────────────────────────────────

class ConstrainPaxisIp(_ProfileBase):
    """
    Fix pressure on magnetic axis (p_axis) and total plasma current (Ip).

    Parameters
    ----------
    p_axis  : float  — pressure on axis [Pa]
    Ip      : float  — total plasma current [A]
    fvac    : float  — vacuum R·Bφ [T·m]
    alpha_m : float  — current shape exponent (default 1.0)
    alpha_n : float  — current shape exponent (default 2.0)
    Raxis   : float  — reference major radius [m] (default 1.0)
    """
    def __init__(self, p_axis, Ip, fvac, alpha_m=1.0, alpha_n=2.0, Raxis=1.0):
        self.p_axis  = p_axis
        self.Ip      = Ip
        self._fvac   = fvac
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n
        self.Raxis   = Raxis
        self.L = self.Beta0 = self.psi_axis = self.psi_bndry = 0.0

    def Jtor(self, R, Z, psi, psi_axis, psi_bndry, mask=None):
        """Compute toroidal current density [A/m²]."""
        dR   = R[1, 0] - R[0, 0]
        dZ   = Z[0, 1] - Z[0, 0]
        dpsi = psi_bndry - psi_axis
        if abs(dpsi) < 1e-30:
            return np.zeros_like(psi)

        psiN      = (psi - psi_axis) / dpsi
        jtorshape = self._shape(psiN)
        if mask is None:
            mask = np.where((psiN >= 0) & (psiN <= 1), 1.0, 0.0)
        jtorshape *= mask

        # Integral of shape function over [0,1] in normalised psi
        shapeintegral, _ = quad(
            lambda x: (1.0 - x**self.alpha_m)**self.alpha_n, 0.0, 1.0)
        shapeintegral *= dpsi

        # Constraint: p_axis = -(L*Beta0/Raxis) * shapeintegral
        LBeta0 = -self.p_axis * self.Raxis / shapeintegral

        # Constraint: Ip = L*Beta0*IR + L*(1-Beta0)*I_R
        IR  = romb(romb(jtorshape * R / self.Raxis)) * dR * dZ
        I_R = romb(romb(jtorshape * self.Raxis / R)) * dR * dZ

        L     = (self.Ip - LBeta0 * (IR - I_R)) / (I_R + 1e-30)
        Beta0 = LBeta0 / (L + 1e-30)

        Jtor = L * (Beta0 * R / self.Raxis
                    + (1.0 - Beta0) * self.Raxis / R) * jtorshape

        self.L = L;  self.Beta0 = Beta0
        self.psi_axis = psi_axis;  self.psi_bndry = psi_bndry
        return Jtor


# ─────────────────────────────────────────────────────────────────────────────
#  ConstrainBetapIp  — fix poloidal beta + Ip
# ─────────────────────────────────────────────────────────────────────────────

class ConstrainBetapIp(_ProfileBase):
    """
    Fix poloidal beta (betap) and total plasma current (Ip).
    Follows Jeon (2015) and FreeGS ConstrainBetapIp.

    βp = 2μ₀ ∫p R dRdZ / (∫ Bpol² R dRdZ)
    """
    def __init__(self, betap, Ip, fvac, alpha_m=1.0, alpha_n=2.0, Raxis=1.0):
        self.betap   = betap
        self.Ip      = Ip
        self._fvac   = fvac
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n
        self.Raxis   = Raxis
        self.L = self.Beta0 = self.psi_axis = self.psi_bndry = 0.0

    def Jtor(self, R, Z, psi, psi_axis, psi_bndry, mask=None,
             Br_2d=None, Bz_2d=None):
        """
        Compute Jtor satisfying βp and Ip constraints.
        Br_2d, Bz_2d must be supplied (or will be estimated from psi).
        """
        dR   = R[1, 0] - R[0, 0]
        dZ   = Z[0, 1] - Z[0, 0]
        dpsi = psi_bndry - psi_axis
        if abs(dpsi) < 1e-30:
            return np.zeros_like(psi)

        psiN      = (psi - psi_axis) / dpsi
        jtorshape = self._shape(psiN)
        if mask is None:
            mask = np.where((psiN >= 0) & (psiN <= 1), 1.0, 0.0)
        jtorshape *= mask

        # Pressure shape integral (normalised)
        def pshape_fn(pn):
            v, _ = quad(lambda x: (1 - x**self.alpha_m)**self.alpha_n, pn, 1.0)
            return v * dpsi

        nx, ny = psiN.shape
        pfunc = np.zeros((nx, ny))
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                if 0.0 <= psiN[i,j] < 1.0:
                    pfunc[i,j] = pshape_fn(float(psiN[i,j]))
        if mask is not None:
            pfunc *= mask

        # βp constraint → LBeta0
        if Br_2d is None or Bz_2d is None:
            # Crude estimate: skip Bpol integral, use simplified form
            p_int  = romb(romb(pfunc * R)) * dR * dZ
            # Approximate Bpol² ≈ (μ₀ Ip / (2π a))² at minor radius a
            from scipy.interpolate import RectBivariateSpline
            f_psi = RectBivariateSpline(R[:,0], Z[0,:], psi)
            Br_2d = -f_psi(R[:,0], Z[0,:], dy=1) / R
            Bz_2d =  f_psi(R[:,0], Z[0,:], dx=1) / R
        B2_int = romb(romb((Br_2d**2 + Bz_2d**2) * mask * R)) * dR * dZ

        LBeta0 = -self.betap * self.Raxis * B2_int / (2.0 * MU0 * romb(romb(pfunc*R))*dR*dZ + 1e-30)

        IR  = romb(romb(jtorshape * R / self.Raxis)) * dR * dZ
        I_R = romb(romb(jtorshape * self.Raxis / R)) * dR * dZ

        L     = (self.Ip - LBeta0 * (IR - I_R)) / (I_R + 1e-30)
        Beta0 = LBeta0 / (L + 1e-30)

        Jtor = L * (Beta0 * R / self.Raxis
                    + (1.0 - Beta0) * self.Raxis / R) * jtorshape

        self.L = L;  self.Beta0 = Beta0
        self.psi_axis = psi_axis;  self.psi_bndry = psi_bndry
        return Jtor


# ─────────────────────────────────────────────────────────────────────────────
#  ConstrainRotation  — fix p₀, Ip, and rigid rotation Ω(ψ)
# ─────────────────────────────────────────────────────────────────────────────

class ConstrainRotation(ConstrainPaxisIp):
    """
    Extend ConstrainPaxisIp with toroidal rotation Ω(ψN) [rad/s].

    The effective pressure includes the centrifugal term:
        p_eff(R, ψ) = p(ψ) + ½ ρ(ψ) Ω²(ψ) R²

    The current profile shape is modified accordingly:
        dp_eff/dψ = p'(ψ) + ρ'(ψ) Ω² R²/2 + ρ(ψ) Ω Ω' R²

    Parameters
    ----------
    omega_profile  : callable  Ω(psiN) [rad/s]
    rho_profile    : callable  ρ(psiN) [kg/m³]  (default: flat ρ₀=1e-7)
    """
    def __init__(self, p_axis, Ip, fvac,
                 omega_profile=None, rho_profile=None,
                 alpha_m=1.0, alpha_n=2.0, Raxis=1.0):
        super().__init__(p_axis, Ip, fvac, alpha_m, alpha_n, Raxis)
        self.omega = omega_profile or (lambda psiN: np.zeros_like(np.asarray(psiN)))
        self.rho   = rho_profile   or (lambda psiN: np.full_like(np.asarray(psiN), 1e-7))

    def Jtor(self, R, Z, psi, psi_axis, psi_bndry, mask=None):
        """Jtor with rotation correction to the effective pressure gradient."""
        dR   = R[1, 0] - R[0, 0]
        dZ   = Z[0, 1] - Z[0, 0]
        dpsi = psi_bndry - psi_axis
        if abs(dpsi) < 1e-30:
            return np.zeros_like(psi)

        psiN      = (psi - psi_axis) / dpsi
        jtorshape = self._shape(psiN)
        if mask is None:
            mask = np.where((psiN >= 0) & (psiN <= 1), 1.0, 0.0)
        jtorshape *= mask

        # Rotation correction to the pressure-like source term
        Omega  = np.vectorize(self.omega)(np.clip(psiN, 0, 1))
        rho_2d = np.vectorize(self.rho)(np.clip(psiN, 0, 1))
        # Effective additional Jtor from rotation: ρ Ω² R
        Jrot = rho_2d * Omega**2 * R * mask

        # Solve for (L, Beta0) using base class logic on jtorshape
        shapeintegral, _ = quad(
            lambda x: (1.0 - x**self.alpha_m)**self.alpha_n, 0.0, 1.0)
        shapeintegral *= dpsi
        LBeta0 = -self.p_axis * self.Raxis / shapeintegral

        IR  = romb(romb(jtorshape * R / self.Raxis)) * dR * dZ
        I_R = romb(romb(jtorshape * self.Raxis / R)) * dR * dZ
        L     = (self.Ip - LBeta0 * (IR - I_R)) / (I_R + 1e-30)
        Beta0 = LBeta0 / (L + 1e-30)

        Jtor = (L * (Beta0 * R / self.Raxis
                     + (1.0 - Beta0) * self.Raxis / R) * jtorshape
                + Jrot)

        self.L = L;  self.Beta0 = Beta0
        self.psi_axis = psi_axis;  self.psi_bndry = psi_bndry
        return Jtor

    def pprime(self, psiN):
        """p'(ψN) including rotation contribution (evaluated at mean R)."""
        base = super().pprime(psiN)
        Omega = float(np.asarray(self.omega(psiN)).flat[0]) if np.isscalar(psiN) else 0.0
        rho   = float(np.asarray(self.rho(psiN)).flat[0])   if np.isscalar(psiN) else 0.0
        return base + rho * Omega**2 * self.Raxis


# ─────────────────────────────────────────────────────────────────────────────
#  ProfilesPprimeFfprime  — specify p'(ψN) and ff'(ψN) directly
# ─────────────────────────────────────────────────────────────────────────────

class ProfilesPprimeFfprime:
    """
    Arbitrary profiles specified by p'(ψN) and ff'(ψN) callables.

    Jtor = R p'(ψN) + ff'(ψN) / (μ₀ R)
    """
    def __init__(self, pprime_fn, ffprime_fn, fvac):
        self._pprime  = pprime_fn
        self._ffprime = ffprime_fn
        self._fvac    = fvac
        self.psi_axis = self.psi_bndry = 0.0

    def pprime(self, psiN):
        return self._pprime(psiN)

    def ffprime(self, psiN):
        return self._ffprime(psiN)

    def fvac(self):
        return self._fvac

    def Jtor(self, R, Z, psi, psi_axis, psi_bndry, mask=None):
        dpsi = psi_bndry - psi_axis
        if abs(dpsi) < 1e-30:
            return np.zeros_like(psi)
        psiN = np.clip((psi - psi_axis) / dpsi, 0.0, 1.0)
        Jtor = R * np.vectorize(self._pprime)(psiN) \
             + np.vectorize(self._ffprime)(psiN) / (MU0 * R)
        if mask is not None:
            Jtor *= mask
        self.psi_axis = psi_axis;  self.psi_bndry = psi_bndry
        return Jtor

    def pressure(self, psiN):
        psiN = np.asarray(psiN, dtype=float)
        scalar = psiN.ndim == 0
        psiN = np.atleast_1d(psiN)
        dpsi = self.psi_bndry - self.psi_axis
        result = np.array([
            quad(self._pprime, float(p), 1.0)[0] * dpsi
            for p in psiN.ravel()
        ]).reshape(psiN.shape)
        return float(result[0]) if scalar else result

    def fpol(self, psiN):
        psiN = np.asarray(psiN, dtype=float)
        scalar = psiN.ndim == 0
        psiN = np.atleast_1d(psiN)
        dpsi = self.psi_bndry - self.psi_axis
        f2vac = self.fvac()**2
        result = np.array([
            np.sqrt(max(2.0*quad(self._ffprime, float(p), 1.0)[0]*dpsi + f2vac, 0.0))
            for p in psiN.ravel()
        ]).reshape(psiN.shape)
        return float(result[0]) if scalar else result
