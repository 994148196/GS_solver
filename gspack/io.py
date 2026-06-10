"""
gspack.io — HDF5 equilibrium archive
"""
import numpy as np


def save(eq, filename):
    try: import h5py
    except ImportError: raise ImportError("pip install h5py")
    from .backend import to_numpy

    psiN_1d = np.linspace(0.01, 0.99, 100)
    sep = eq.separatrix(npoints=360)

    with h5py.File(filename, 'w') as f:
        g = f.create_group('grid')
        g.create_dataset('R', data=to_numpy(eq.R))
        g.create_dataset('Z', data=to_numpy(eq.Z))
        for k in ['Rmin','Rmax','Zmin','Zmax','nx','ny']:
            g.attrs[k] = getattr(eq, k)

        fld = f.create_group('fields')
        fld.create_dataset('psi',        data=to_numpy(eq.psi()))
        fld.create_dataset('plasma_psi', data=to_numpy(eq.plasma_psi))
        fld.create_dataset('psiN',       data=to_numpy(eq.psiN()))
        fld.create_dataset('Jtor',       data=to_numpy(eq.Jtor))
        fld.create_dataset('Bpol',       data=to_numpy(eq.Bpol()))

        sc = f.create_group('scalars')
        sc.attrs['psi_axis']  = float(eq.psi_axis)
        sc.attrs['psi_bndry'] = float(eq.psi_bndry)
        sc.attrs['Ip']        = float(eq.plasmaCurrent())
        R_m, Z_m, _ = eq.magneticAxis()
        sc.attrs['R_mag'] = float(R_m);  sc.attrs['Z_mag'] = float(Z_m)
        geo = eq.geometricAxis()
        sc.attrs['R_geo'] = float(geo[0]);  sc.attrs['Z_geo'] = float(geo[1])
        sc.attrs['kappa'] = float(eq.elongation())
        sc.attrs['delta'] = float(eq.triangularity())
        sc.attrs['a']     = float(eq.minorRadius())
        sc.attrs['betap'] = float(eq.poloidalBeta())
        sc.attrs['volume']= float(eq.plasmaVolume())
        Ri, Ro = eq.innerOuterSeparatrix()
        sc.attrs['R_inner'] = float(Ri);  sc.attrs['R_outer'] = float(Ro)

        prof = f.create_group('profiles')
        prof.create_dataset('psiN_1d', data=psiN_1d)
        if eq._profiles is not None:
            prof.create_dataset('q',       data=eq.q(psiN_1d))
            prof.create_dataset('pressure',data=np.array([eq._profiles.pressure(p) for p in psiN_1d]))
            prof.create_dataset('fpol',    data=np.array([eq._profiles.fpol(p)     for p in psiN_1d]))

        sg = f.create_group('separatrix')
        sg.create_dataset('R', data=sep[:,0]);  sg.create_dataset('Z', data=sep[:,1])

        cg = f.create_group('coils')
        for name, coil in eq.tokamak.coils:
            cg.attrs[name] = float(coil.current)

        if eq._opoints:
            f.create_dataset('opoints', data=np.array(eq._opoints))
        if eq._xpoints:
            f.create_dataset('xpoints', data=np.array(eq._xpoints))

    print(f"  Saved → {filename}")


def load(filename):
    try: import h5py
    except ImportError: raise ImportError("pip install h5py")
    out = {}
    with h5py.File(filename, 'r') as f:
        out['R'] = f['grid/R'][:];  out['Z'] = f['grid/Z'][:]
        for k in ['Rmin','Rmax','Zmin','Zmax','nx','ny']:
            out[k] = f['grid'].attrs[k]
        for k in ['psi','plasma_psi','psiN','Jtor','Bpol']:
            if f'fields/{k}' in f: out[k] = f[f'fields/{k}'][:]
        for k,v in f['scalars'].attrs.items(): out[k] = float(v)
        if 'profiles' in f:
            out['psiN_1d'] = f['profiles/psiN_1d'][:]
            for k in ['q','pressure','fpol']:
                if f'profiles/{k}' in f: out[k] = f[f'profiles/{k}'][:]
        if 'separatrix' in f:
            out['sep_R'] = f['separatrix/R'][:]
            out['sep_Z'] = f['separatrix/Z'][:]
        out['coils'] = dict(f['coils'].attrs)
        if 'opoints' in f: out['opoints'] = f['opoints'][:]
        if 'xpoints' in f: out['xpoints'] = f['xpoints'][:]
    # Convert dict to object with attribute access
    class _LoadedEquil:
        def __init__(self, d):
            self.__dict__.update(d)
        def __repr__(self):
            return f"LoadedEquil(Ip={self.__dict__.get('Ip',0):.3e})"
    return _LoadedEquil(out)
