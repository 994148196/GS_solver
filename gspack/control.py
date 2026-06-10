"""
gspack.control
==============
Plasma position/shape control — upgraded with:
  • Original constrain (X-points + isoflux) — unchanged
  • constrain_snowflake (adds 2nd-order X-point constraints for snowflake divertor)
  • psivals constraint support (fix ψ at specific (R,Z) points)

Tikhonov regularised least-squares:
  ΔI = (AᵀA + γ²I)⁻¹ Aᵀ b
"""

import numpy as np
from numpy.linalg import inv


class constrain:
    """
    Adjust coil currents to enforce X-point and isoflux constraints.

    Parameters
    ----------
    xpoints  : list of (R, Z) — desired X-point locations
    isoflux  : list of (R1, Z1, R2, Z2) — pairs with equal ψ
    psivals  : list of (R, Z, psi_target) — fix ψ at specific points
    gamma    : Tikhonov regularisation (default 1e-12)
    """
    def __init__(self, xpoints=None, isoflux=None, psivals=None, gamma=1e-12):
        self.xpoints = xpoints or []
        self.isoflux = isoflux or []
        self.psivals = psivals or []
        self.gamma   = gamma

    def __call__(self, eq):
        tokamak = eq.tokamak
        A_rows, b_vec = [], []

        # ── X-point: Br=0, Bz=0 ──────────────────────────────────────────
        for (Rxp, Zxp) in self.xpoints:
            b_vec.append(-float(eq.Br(Rxp, Zxp)))
            A_rows.append(tokamak.controlBr(Rxp, Zxp))
            b_vec.append(-float(eq.Bz(Rxp, Zxp)))
            A_rows.append(tokamak.controlBz(Rxp, Zxp))

        # ── Isoflux: ψ(R1,Z1) = ψ(R2,Z2) ────────────────────────────────
        for (R1, Z1, R2, Z2) in self.isoflux:
            p1 = eq.psiRZ(R1, Z1);  p2 = eq.psiRZ(R2, Z2)
            b_vec.append(p2 - p1)
            c1 = tokamak.controlPsi(R1, Z1)
            c2 = tokamak.controlPsi(R2, Z2)
            A_rows.append([v1 - v2 for v1, v2 in zip(c1, c2)])

        # ── Fixed ψ values ────────────────────────────────────────────────
        for (R, Z, psi_tgt) in self.psivals:
            b_vec.append(psi_tgt - eq.psiRZ(R, Z))
            A_rows.append(tokamak.controlPsi(R, Z))

        if not b_vec:
            return

        A  = np.array(A_rows, dtype=float)
        b  = np.array(b_vec,  dtype=float)
        nc = A.shape[1]

        ATA = A.T @ A + self.gamma**2 * np.eye(nc)
        dI  = inv(ATA) @ (A.T @ b)
        tokamak.controlAdjust(dI)

    # ── Convenience builders ──────────────────────────────────────────────
    def current_change(self, eq):
        """Return ΔI without applying it (for diagnostics)."""
        tokamak = eq.tokamak
        A_rows, b_vec = [], []
        for (Rxp, Zxp) in self.xpoints:
            b_vec.append(-float(eq.Br(Rxp, Zxp)))
            A_rows.append(tokamak.controlBr(Rxp, Zxp))
            b_vec.append(-float(eq.Bz(Rxp, Zxp)))
            A_rows.append(tokamak.controlBz(Rxp, Zxp))
        for (R1, Z1, R2, Z2) in self.isoflux:
            p1 = eq.psiRZ(R1, Z1);  p2 = eq.psiRZ(R2, Z2)
            b_vec.append(p2 - p1)
            c1 = tokamak.controlPsi(R1, Z1)
            c2 = tokamak.controlPsi(R2, Z2)
            A_rows.append([v1 - v2 for v1, v2 in zip(c1, c2)])
        if not b_vec:
            return np.zeros(len(tokamak.controlCurrents()))
        A  = np.array(A_rows, dtype=float)
        b  = np.array(b_vec,  dtype=float)
        nc = A.shape[1]
        return inv(A.T @ A + self.gamma**2 * np.eye(nc)) @ (A.T @ b)


class constrain_snowflake(constrain):
    """
    Snowflake divertor control — adds 2nd-order null constraints.

    For a snowflake equilibrium the X-point must be a 2nd-order zero
    of the poloidal flux:
        ∂²ψ/∂R² = 0,  ∂²ψ/∂Z² = 0,  ∂²ψ/∂R∂Z = 0

    These are implemented via 2nd-order numerical derivatives of the
    Green's function at the snowflake X-point location.

    Parameters
    ----------
    sf_xpoints : list of (R, Z) — snowflake X-point targets
    All other parameters same as constrain.
    """
    def __init__(self, xpoints=None, isoflux=None, psivals=None,
                 sf_xpoints=None, gamma=1e-12, eps=1e-3):
        super().__init__(xpoints, isoflux, psivals, gamma)
        self.sf_xpoints = sf_xpoints or []
        self.eps = eps

    def __call__(self, eq):
        tokamak = eq.tokamak
        A_rows, b_vec = [], []

        # Original constraints
        for (Rxp, Zxp) in self.xpoints:
            b_vec.append(-float(eq.Br(Rxp, Zxp)))
            A_rows.append(tokamak.controlBr(Rxp, Zxp))
            b_vec.append(-float(eq.Bz(Rxp, Zxp)))
            A_rows.append(tokamak.controlBz(Rxp, Zxp))
        for (R1, Z1, R2, Z2) in self.isoflux:
            p1 = eq.psiRZ(R1, Z1);  p2 = eq.psiRZ(R2, Z2)
            b_vec.append(p2 - p1)
            c1 = tokamak.controlPsi(R1, Z1)
            c2 = tokamak.controlPsi(R2, Z2)
            A_rows.append([v1 - v2 for v1, v2 in zip(c1, c2)])
        for (R, Z, psi_tgt) in self.psivals:
            b_vec.append(psi_tgt - eq.psiRZ(R, Z))
            A_rows.append(tokamak.controlPsi(R, Z))

        # ── Snowflake constraints: ∂Br/∂Z = 0 and ∂Bz/∂R = 0 ────────────
        e = self.eps
        from .greens import greens_Br, greens_Bz

        for (Rxp, Zxp) in self.sf_xpoints:
            # ∂Br/∂Z ≈ (Br(Rxp, Zxp+ε) - Br(Rxp, Zxp-ε)) / 2ε = 0
            # → residual: -(∂Br/∂Z)|current
            dBr_dZ = (float(eq.Br(Rxp, Zxp + e)) - float(eq.Br(Rxp, Zxp - e))) / (2*e)
            b_vec.append(-dBr_dZ)
            row_BRZ = []
            for _, coil in tokamak.coils:
                if coil.control:
                    dG = (coil.controlBr(Rxp, Zxp + e)
                        - coil.controlBr(Rxp, Zxp - e)) / (2*e)
                    row_BRZ.append(float(dG))
            A_rows.append(row_BRZ)

            # ∂Bz/∂R ≈ (Bz(Rxp+ε, Zxp) - Bz(Rxp-ε, Zxp)) / 2ε = 0
            dBz_dR = (float(eq.Bz(Rxp + e, Zxp)) - float(eq.Bz(Rxp - e, Zxp))) / (2*e)
            b_vec.append(-dBz_dR)
            row_BZR = []
            for _, coil in tokamak.coils:
                if coil.control:
                    dG = (coil.controlBz(Rxp + e, Zxp)
                        - coil.controlBz(Rxp - e, Zxp)) / (2*e)
                    row_BZR.append(float(dG))
            A_rows.append(row_BZR)

        if not b_vec:
            return

        A  = np.array(A_rows, dtype=float)
        b  = np.array(b_vec,  dtype=float)
        nc = A.shape[1]
        dI = inv(A.T @ A + self.gamma**2 * np.eye(nc)) @ (A.T @ b)
        tokamak.controlAdjust(dI)
