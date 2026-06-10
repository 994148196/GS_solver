"""
gspack.geqdsk — G-EQDSK read/write (COCOS 1)
Compatible with TRANSP, VMEC, GENE, BOUT++, OMFIT
"""
import numpy as np
import re
from datetime import datetime


def _wline(f, vals):
    vals = list(vals)
    while vals:
        f.write("".join(f"{v:16.9e}" for v in vals[:5]) + "\n")
        vals = vals[5:]


def write_geqdsk(eq, filename, shot=0, time=0.0):
    import numpy as _np
    nw, nh = eq.nx, eq.ny
    psiN_1d = _np.linspace(0.0, 1.0, nw)
    R_mag, Z_mag, _ = eq.magneticAxis()
    R_geo, Z_geo    = eq.geometricAxis()
    fvac  = eq._profiles.fvac() if eq._profiles else 1.0
    Bt0   = fvac / (R_mag + 1e-30)
    Ip    = eq.plasmaCurrent()
    ts    = datetime.now().strftime("%m/%d/%Y")

    with open(filename, 'w') as f:
        f.write(f"  gspack  {ts}  #{shot:05d}  t={time:.4f}s   {nw:4d}  {nh:4d}\n")
        _wline(f, [eq.Rmax-eq.Rmin, eq.Zmax-eq.Zmin, R_geo, eq.Rmin, 0.5*(eq.Zmin+eq.Zmax)])
        _wline(f, [R_mag, Z_mag, eq.psi_axis, eq.psi_bndry, Bt0])
        _wline(f, [Ip, eq.psi_axis, 0.0, R_mag, 0.0])
        _wline(f, [Z_mag, 0.0, eq.psi_bndry, 0.0, 0.0])
        f_p  = _np.array([eq._profiles.fpol(p)     for p in psiN_1d])
        pr_p = _np.array([eq._profiles.pressure(p) for p in psiN_1d])
        dpsi = eq.psi_axis - eq.psi_bndry
        ffp  = _np.gradient(0.5*f_p**2, -dpsi*psiN_1d + eq.psi_axis)
        pp   = _np.gradient(pr_p,       -dpsi*psiN_1d + eq.psi_axis)
        _wline(f, f_p);  _wline(f, pr_p);  _wline(f, ffp);  _wline(f, pp)
        _wline(f, eq.psi().T.ravel())
        _wline(f, eq.q(psiN_1d))
        sep = eq.separatrix(npoints=nw)
        f.write(f"{len(sep):5d}   0\n")
        _wline(f, sep.T.ravel())
    print(f"  G-EQDSK → {filename}")



class EqdskData:
    """Container for G-EQDSK data with attribute access."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
    def __repr__(self):
        return f"EqdskData(nw={self.nw}, nh={self.nh}, Ip={getattr(self,'current',0):.3e})"


def read_geqdsk(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    ints = re.findall(r'\b(\d+)\b', lines[0])
    nw, nh = int(ints[-2]), int(ints[-1])
    body = ''.join(lines[1:])
    nums = list(map(float, re.findall(r'[-+]?\d+\.\d+[Ee][+-]\d+', body)))
    RDIM,ZDIM,RCENTR,RLEFT,ZMID = nums[0:5]
    RMAXIS,ZMAXIS,SIMAG,SIBRY,BCENTR = nums[5:10]
    CURRENT = nums[10]
    n1d, n2d = nw, nw*nh
    off = 20
    fpol   = np.array(nums[off:off+n1d])
    pres   = np.array(nums[off+n1d:off+2*n1d])
    ffprim = np.array(nums[off+2*n1d:off+3*n1d])
    pprime = np.array(nums[off+3*n1d:off+4*n1d])
    psi_2d = np.array(nums[off+4*n1d:off+4*n1d+n2d]).reshape(nh,nw).T
    qpsi   = np.array(nums[off+4*n1d+n2d:off+5*n1d+n2d])
    R1d = np.linspace(RLEFT, RLEFT+RDIM, nw)
    Z1d = np.linspace(ZMID-ZDIM/2, ZMID+ZDIM/2, nh)
    R, Z = np.meshgrid(R1d, Z1d, indexing='ij')
    return EqdskData({'nw':nw,'nh':nh,'R':R,'Z':Z,'psi':psi_2d,
            'psi_axis':SIMAG,'psi_bndry':SIBRY,
            'fpol':fpol,'pressure':pres,'ffprim':ffprim,'pprime':pprime,'q':qpsi,
            'scalars':{'RMAXIS':RMAXIS,'ZMAXIS':ZMAXIS,'BCENTR':BCENTR,'CURRENT':CURRENT,
                       'RDIM':RDIM,'ZDIM':ZDIM,'RLEFT':RLEFT,'ZMID':ZMID,'RMAXIS':RMAXIS,'ZMAXIS':ZMAXIS,'BCENTR':BCENTR,'CURRENT':CURRENT}})

# Aliases
write = write_geqdsk
read  = read_geqdsk
