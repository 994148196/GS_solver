"""
gspack.machine
==============
Tokamak coil definitions — GPU transparent.

Key fix: coil.R and coil.Z are stored as plain Python floats.
When psi/Br/Bz are called with CuPy arrays for R, Z, the scalar
source coordinates are converted to the active backend via asarray()
inside greens(), so no manual conversion is needed here.
"""

import numpy as np
from .backend import to_numpy, to_backend, asarray, get_xp
from .greens  import greens, greens_Br, greens_Bz


class Coil:
    """Single filament coil at (R, Z)."""
    def __init__(self, R, Z, current=0.0, turns=1, control=True):
        self.R       = float(R)
        self.Z       = float(Z)
        self.current = float(current)
        self.turns   = int(turns)
        self.control = bool(control)

    def psi(self, R, Z):
        return greens(self.R, self.Z, R, Z) * self.current * self.turns

    def Br(self, R, Z):
        return greens_Br(self.R, self.Z, R, Z) * self.current * self.turns

    def Bz(self, R, Z):
        return greens_Bz(self.R, self.Z, R, Z) * self.current * self.turns

    def controlPsi(self, R, Z):
        return float(to_numpy(greens(self.R, self.Z,
                                     float(R), float(Z))).flat[0]) * self.turns

    def controlBr(self, R, Z):
        return float(to_numpy(greens_Br(self.R, self.Z,
                                        float(R), float(Z))).flat[0]) * self.turns

    def controlBz(self, R, Z):
        return float(to_numpy(greens_Bz(self.R, self.Z,
                                        float(R), float(Z))).flat[0]) * self.turns

    def __repr__(self):
        return f"Coil(R={self.R}, Z={self.Z}, I={self.current:.3g} A)"


class ShapedCoil:
    """
    Rectangular cross-section coil, approximated by 4 corner filaments.

    corners : list of (R, Z) tuples
    """
    def __init__(self, corners, current=0.0, turns=1, control=True):
        self.corners = [(float(r), float(z)) for r, z in corners]
        self.current = float(current)
        self.turns   = int(turns)
        self.control = bool(control)
        Rs = [r for r, z in corners]
        Zs = [z for r, z in corners]
        self.R = float(np.mean(Rs))
        self.Z = float(np.mean(Zs))
        self._filaments = [Coil(r, z, 1.0, 1) for r, z in corners]
        self._w = 1.0 / len(corners)

    def _apply(self, method, R, Z):
        result = None
        for f in self._filaments:
            v = getattr(f, method)(R, Z) * self._w
            result = v if result is None else result + v
        return result

    def psi(self, R, Z):
        return self._apply("psi", R, Z) * self.current * self.turns

    def Br(self, R, Z):
        return self._apply("Br", R, Z) * self.current * self.turns

    def Bz(self, R, Z):
        return self._apply("Bz", R, Z) * self.current * self.turns

    def controlPsi(self, R, Z):
        R, Z = float(R), float(Z)
        return sum(float(to_numpy(greens(r, z, R, Z)).flat[0]) * self._w
                   for r, z in self.corners) * self.turns

    def controlBr(self, R, Z):
        R, Z = float(R), float(Z)
        return sum(float(to_numpy(greens_Br(r, z, R, Z)).flat[0]) * self._w
                   for r, z in self.corners) * self.turns

    def controlBz(self, R, Z):
        R, Z = float(R), float(Z)
        return sum(float(to_numpy(greens_Bz(r, z, R, Z)).flat[0]) * self._w
                   for r, z in self.corners) * self.turns

    def __repr__(self):
        return f"ShapedCoil(corners={self.corners}, I={self.current:.3g} A)"


class Wall:
    def __init__(self, R, Z):
        self.R = list(R)
        self.Z = list(Z)


class Machine:
    def __init__(self, coils, wall=None):
        self.coils = coils
        self.wall  = wall

    def psi_coils(self, R, Z):
        xp = get_xp()
        result = xp.zeros_like(asarray(R))
        for _, coil in self.coils:
            result = result + coil.psi(R, Z)
        return result

    def Br_coils(self, R, Z):
        xp = get_xp()
        result = xp.zeros_like(asarray(R))
        for _, coil in self.coils:
            result = result + coil.Br(R, Z)
        return result

    def Bz_coils(self, R, Z):
        xp = get_xp()
        result = xp.zeros_like(asarray(R))
        for _, coil in self.coils:
            result = result + coil.Bz(R, Z)
        return result

    def controlCurrents(self):
        return [coil.current for _, coil in self.coils if coil.control]

    def setControlCurrents(self, currents):
        idx = 0
        for _, coil in self.coils:
            if coil.control:
                coil.current = float(currents[idx])
                idx += 1

    def controlAdjust(self, delta_currents):
        idx = 0
        for _, coil in self.coils:
            if coil.control:
                coil.current += float(delta_currents[idx])
                idx += 1

    def controlPsi(self, R, Z):
        return [coil.controlPsi(R, Z)
                for _, coil in self.coils if coil.control]

    def controlBr(self, R, Z):
        return [coil.controlBr(R, Z)
                for _, coil in self.coils if coil.control]

    def controlBz(self, R, Z):
        return [coil.controlBz(R, Z)
                for _, coil in self.coils if coil.control]

    def printCurrents(self):
        print("=" * 26)
        for name, coil in self.coils:
            print(f"  {name:5s}  {coil.current:+.1f} A")
        print("=" * 26)


def TestTokamak():
    coils = [
        ("P1L", ShapedCoil([(0.95,-1.15),(0.95,-1.05),(1.05,-1.05),(1.05,-1.15)])),
        ("P1U", ShapedCoil([(0.95, 1.15),(0.95, 1.05),(1.05, 1.05),(1.05, 1.15)])),
        ("P2L", Coil(1.75, -0.6)),
        ("P2U", Coil(1.75,  0.6)),
    ]
    wall = Wall(
        R=[0.75, 0.75, 1.50, 1.80, 1.80, 1.50],
        Z=[-0.85, 0.85, 0.85, 0.25, -0.25, -0.85],
    )
    return Machine(coils, wall)
