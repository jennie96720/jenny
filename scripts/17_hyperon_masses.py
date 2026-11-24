"""Lambda and Sigma^0 (and the proton) from the flavour-resolved quark+diquark Faddeev equation
(qdq/baryon_faddeev.py), strange quark of Chen, Krein, Roberts, Schmidt, arXiv:1901.04305 App. A.

Checks printed:
  * proton flavour factors c_ss, c_sa, c_aa = 1/2, -sqrt3/2, -1/2; couplings equal those of qdq/diquark.py;
  * proton lambda(1.18) on the reference grid (8,4,4) = 1.0154017190 (docs/reference_values.md);
  * SU(3) limit (m_s = m_u, equal diquark masses and couplings): Lambda and Sigma^0 have the nucleon's
    eigenvalue as their largest one;
  * setup C = diquark masses of 1901.04305 (0.80, 0.90, 0.95, 1.05, 1.20 GeV): canonical couplings
    reproduce their Eq. (7) (2|g| = 14.75, 9.45, 12.73, 7.62, 3.72), core masses vs their Table I
    (N 1.19(13), Lambda 1.37(14), Sigma 1.41(14));
  * setup A = 2611.19819 diquark masses 0.79, 0.89 GeV + 0.15 GeV per s quark (equal-spacing rule).
Grid convergence for setup A: (nr, ncheb) = (16, 6) vs (24, 8).

Usage: python scripts/17_hyperon_masses.py  -> results/hyperon_masses.json
"""
import sys, json, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qdq.baryon_faddeev import BaryonFaddeev, BARYONS, make_group, flavour_coeffs, diquark_coupling, QUARKS

SETUPS = {
    'C': {('s', 'll'): 0.80, ('a', 'll'): 0.90, ('s', 'ls'): 0.95, ('a', 'ls'): 1.05, ('a', 'ss'): 1.20},
    'A': {('s', 'll'): 0.79, ('a', 'll'): 0.89, ('s', 'ls'): 0.94, ('a', 'ls'): 1.04, ('a', 'ss'): 1.19},
}
STARTS = {'p': (1.15, 1.20), 'Lambda': (1.30, 1.40), 'Sigma0': (1.35, 1.45)}


def couplings(ms):
    g = {}
    for (J, dq), m in ms.items():
        qa = QUARKS['l'] if dq[0] == 'l' else QUARKS['s']
        qb = QUARKS['l'] if dq[1] == 'l' else QUARKS['s']
        g[J, dq] = diquark_coupling(J, m, qa, qb)[0]
    return g


def main():
    t0 = time.time()
    out = {}
    for b in BARYONS:
        c = flavour_coeffs([make_group(*gr) for gr in BARYONS[b]])
        print(f"flavour factors {b}:", {k: round(float(v), 6) for k, v in c.items()})
    ref = dict(nr=8, nz=4, ny=4, ncheb=None)
    gp = {('s', 'll'): 7.47, ('a', 'll'): 6.49j}
    lam = BaryonFaddeev('p', couplings=gp, **ref).eigen(1.18)[0]
    print(f"proton lambda(1.18), grid (8,4,4): {lam.real:.10f}   (reference 1.0154017190)")
    su3 = dict(quarks={'s': QUARKS['l']}, masses={('s', 'ls'): 0.79, ('a', 'ls'): 0.89, ('a', 'ss'): 0.89},
               couplings={('s', 'll'): 7.47, ('a', 'll'): 6.49j, ('s', 'ls'): 7.47, ('a', 'ls'): 6.49j,
                          ('a', 'ss'): 6.49j})
    out['su3_limit'] = {}
    for b in ('Lambda', 'Sigma0'):
        sp = BaryonFaddeev(b, **su3, **ref).spectrum(1.18, 3)
        out['su3_limit'][b] = sp[0].real
        print(f"SU(3) limit {b:7s}: largest eigenvalues at 1.18 = {np.round(sp, 10)}")
    for name, ms in SETUPS.items():
        g = couplings(ms)
        print(f"setup {name}: 2|g| =", {f'{J}{dq}': round(2 * abs(v), 2) for (J, dq), v in g.items()})
        grids = [dict(nr=16, nz=8, ny=8, ncheb=6)] + ([dict(nr=24, nz=10, ny=8, ncheb=8)] if name == 'A' else [])
        out[name] = dict(masses={f'{J}{dq}': m for (J, dq), m in ms.items()},
                         couplings={f'{J}{dq}': [complex(v).real, complex(v).imag] for (J, dq), v in g.items()},
                         M={})
        for gr in grids:
            tag = f"nr{gr['nr']}_ncheb{gr['ncheb']}"
            out[name]['M'][tag] = {}
            for b, (m0, m1) in STARTS.items():
                M = BaryonFaddeev(b, masses=ms, couplings=g, **gr).solve_mass(m0, m1)
                out[name]['M'][tag][b] = M
                print(f"   setup {name} {tag:14s} {b:7s} M = {M:.4f} GeV   ({time.time()-t0:.0f}s)", flush=True)
    pathlib.Path('results').mkdir(exist_ok=True)
    json.dump(out, open('results/hyperon_masses.json', 'w'), indent=1)
    print("saved results/hyperon_masses.json")


if __name__ == '__main__':
    main()
