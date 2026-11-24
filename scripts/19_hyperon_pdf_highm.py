"""Higher Mellin moments (m <= 16) of the hadron-scale valence DFs of p, Lambda, Sigma^0 (qdq/baryon_pdf.py)
on the converged grids of the proton (scripts/12, fortran/runs/fine):
  Faddeev nr = 32, nz = 10, ny = 8, ncheb = 8; one loop (64, 32, 24); two loop (16,14,10)/(16,14,10,10);
  |k| < 30 GeV; regulator (1 + lam p^2)^-m for lam in LAMS, extrapolated to lam = 0 with the basis
  {1, lam, lam log lam, lam^2} on [0.005, 0.05]; variants [0.005, 0.1] and no lam^2 give the error
  (as scripts/12).  Diagram 2 uses the quark-in-diquark moments of phi^2 (scripts/18 zmoments), applied
  per group and struck flavour after the extrapolation.
Setups (scripts/18): A (central), A10, A20 (diquark mass spacing 0.10, 0.20 GeV per s quark: error band).
The proton is computed for setup A (A10, A20 share it: same light diquarks).
Timing (4 cores): Sigma0 ~40 min, Lambda ~25 min, proton ~20 min.

Usage: python scripts/19_hyperon_pdf_highm.py            # compute (resumable) -> results/hyperon_pdf_highm_raw.json
       python scripts/19_hyperon_pdf_highm.py --analyse  # -> results/hyperon_pdf_highm.json
"""
import sys, json, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import importlib.util
import numpy as np
from qdq.baryon_faddeev import BaryonFaddeev
from qdq.baryon_pdf import BaryonPDFMoments, CLASS

_spec = importlib.util.spec_from_file_location('s18', pathlib.Path(__file__).with_name('18_hyperon_pdf.py'))
s18 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s18)

MM = 16
LAMS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
FINE = dict(nr=32, nz=10, ny=8, ncheb=8)
ONE, OUTER, INNER, KMAX = (64, 32, 24), (16, 14, 10), (16, 14, 10, 10), 30.0
RAW = pathlib.Path('results/hyperon_pdf_highm_raw.json')
OUT = pathlib.Path('results/hyperon_pdf_highm.json')
SETUPS = {'A': s18.SETUP_A, 'A10': s18.spaced(0.10), 'A20': s18.spaced(0.20)}
JOBS = [('A', 'p'), ('A', 'Lambda'), ('A', 'Sigma0'), ('A10', 'Lambda'), ('A10', 'Sigma0'),
        ('A20', 'Lambda'), ('A20', 'Sigma0')]


def compute():
    raw = json.loads(RAW.read_text()) if RAW.exists() else {}
    raw.update(lams=LAMS, mmax=MM, grid='Faddeev nr32 nz10 ny8 ncheb8; one (64,32,24); two (16,14,10)/(16,14,10,10); kmax 30')
    t0 = time.time()
    zcache = {}
    for setup, b in JOBS:
        key = f'{setup}/{b}'
        if key in raw:
            continue
        ms = SETUPS[setup]
        g = s18.couplings(ms)
        if setup not in zcache:
            z, _ = s18.zmoments(ms, mmax=MM)
            zcache[setup] = z
            raw[f'{setup}/z'] = {f'{J}{dq}{c}': v.tolist() for (J, dq, c), v in z.items()}
        m0, m1 = s18.STARTS[b] if setup == 'A' else (s18.STARTS[b][0] - 0.07, s18.STARTS[b][1] + 0.07)
        F = BaryonFaddeev(b, masses=ms, couplings=g, **FINE)
        M = F.solve_mass(m0, m1)
        print(f"{key}: M = {M:.5f}  ({time.time()-t0:.0f}s)", flush=True)
        pm = BaryonPDFMoments(F, one=ONE, outer=OUTER, inner=INNER, mmax=MM, kmax=KMAX, lams=LAMS)
        P = np.array([0, 0, 0, 1j * M])
        nb = pm.norm(P)
        one = pm.one_loop(P)
        two = pm.two_loop(P)
        r = pm.moments(P, z_moments={}, one=one, two=two, nb=nb)        # diagram 2 with z = 2^-m: replaced below
        ent = {'M': M, 'groups': [dict(name=gr['name'], J=gr['J'], dq=gr['dq']) for gr in F.groups]}
        for q in 'uds':
            ent[q] = {k: np.real(r[q][k]).tolist() for k in ('1', '4', '5', '6')}
            ent[q]['y'] = {gn: np.real(v).tolist() for gn, v in r[q]['diquark_y'].items()}
            ent[q]['max_imag'] = float(np.abs(np.imag(r[q]['total'])).max())
        raw[key] = ent
        RAW.write_text(json.dumps(raw))
        print(f"{key}: done, numbers {[round(float(np.real(r[q]['total'][0, 0])), 5) for q in 'uds']}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    return raw


def extrap(lams, v, lo, hi, quad=True):
    sel = (lams >= lo) & (lams <= hi)
    l = lams[sel]
    A = [np.ones_like(l), l, l * np.log(l)] + ([l**2] if quad else [])
    c, *_ = np.linalg.lstsq(np.array(A).T, v[sel], rcond=None)
    return c[0]


def analyse(raw):
    from qdq.evolution import AOEvolution
    lams = np.array(raw['lams'])
    MMr = raw['mmax']
    ev = AOEvolution()
    fac = np.array([1.0] + [ev.valence_factor(n, 2.0) for n in range(1, MMr + 1)])
    out = {'lams': raw['lams'], 'grid': raw['grid'], 'valence_factor_2GeV': fac.tolist()}
    for key, ent in raw.items():
        if '/' not in key or key.endswith('/z'):
            continue
        setup = key.split('/')[0]
        z = {k: np.array(v) for k, v in raw[f'{setup}/z'].items()}
        res = {'M': ent['M']}
        for q in 'uds':
            comp = {k: np.array(ent[q][k]) for k in ('1', '4', '5', '6')}
            comp['2'] = 0
            for gr in ent['groups']:
                zz = z.get(f"{gr['J']}{gr['dq']}{CLASS[q]}", 0.5 ** np.arange(MMr + 1))
                comp['2'] = comp['2'] + np.array(ent[q]['y'][gr['name']]) * zz
            cen, err, parts = 0, 0, {}
            for name, v in comp.items():
                A = np.array([extrap(lams, v[:, m], 0.005, 0.05) for m in range(MMr + 1)])
                B = np.array([extrap(lams, v[:, m], 0.005, 0.10) for m in range(MMr + 1)])
                C = np.array([extrap(lams, v[:, m], 0.005, 0.05, quad=False) for m in range(MMr + 1)])
                parts[name] = A.tolist()
                cen, err = cen + A, err + np.maximum(abs(A - B), abs(A - C))
            if abs(cen[0]) < 1e-12:
                res[q] = None
                continue
            rel = err / abs(cen)
            mono = np.concatenate([[True], np.diff(cen) < 0])
            ok = (rel < 0.05) & np.cumprod(mono).astype(bool)
            mrel = int(np.max(np.nonzero(ok)[0]))
            res[q] = {'moments': cen.tolist(), 'error': err.tolist(), 'm_reliable': mrel, 'parts': parts,
                      'lam0': sum(comp[k][0] for k in comp).tolist(), '2GeV': (cen * fac).tolist()}
        out[key] = res
    return out


def report(out):
    np.set_printoptions(precision=5, suppress=True, linewidth=170)
    for key, res in out.items():
        if '/' not in key:
            continue
        print(f"\n{key}: M = {res['M']:.5f}")
        tot1 = 0
        for q in 'uds':
            o = res[q]
            if o is None:
                continue
            k = o['m_reliable']
            tot1 += o['moments'][1]
            print(f"  <x^m>_{q}, m = 0..{k} (rel. error < 5 %, monotonic):\n    {np.array(o['moments'][:k + 1])}"
                  f"\n  +-{np.array(o['error'][:k + 1])}")
        print(f"  momentum sum (m = 1): {tot1:.5f}")


if __name__ == '__main__':
    raw = json.loads(RAW.read_text()) if '--analyse' in sys.argv else compute()
    out = analyse(raw)
    report(out)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"saved {OUT}")
