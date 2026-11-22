"""PDF: higher Mellin moments of the hadron-scale proton valence DFs (m <= 20 attempted).

Converged numerics (cf. scripts/09): Faddeev amplitude nr = 32, ncheb = 8; one-loop grid (64,32,24);
two-loop grid (16,14,10)/(16,14,10,10) (the old (12,6,6) grid was not converged beyond m ~ 3);
|k| < 30 GeV.  Each weighted insertion carries the regulator (1 + lam p^2)^-m (qdq/pdf.py); results for
lam in [0.005, 0.05] are extrapolated to lam = 0 with the basis {1, lam, lam log lam, lam^2} (validated on
diquark-DA moments known at lam = 0: 0.2-2 %).  Variants (lam up to 0.1; no lam^2 term) give the error.
Diagram 2 uses <z^m>_{q/D} of phi_D^2 (scripts/10).

Usage:  python scripts/12_pdf_high_moments.py            # full computation (~80 min on 4 cores)
        python scripts/12_pdf_high_moments.py --analyse  # re-analyse results/pdf_highm_raw.json
"""
import sys, json, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

MM = 20
LAMS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
RAW = pathlib.Path('results/pdf_highm_raw.json')

def compute():
    from qdq.faddeev import NucleonFaddeev
    from qdq.current import NucleonCurrent
    from qdq.exchange import ExchangeCurrent
    from qdq.pdf import NucleonPDFMoments
    t0 = time.time()
    F = NucleonFaddeev(g=(14.94 / 2, 1j * 12.98 / 2), nr=32, nz=10, ny=8, ncheb=8)
    F.solve_mass(1.17, 1.19, verbose=False)
    P = np.array([0, 0, 0, 1j * F.M])
    J = NucleonCurrent(F, nk=64, nz=32, ny=24)
    X = ExchangeCurrent(J, nk=(16, 14, 10), nl=(16, 14, 10, 10))
    pm = NucleonPDFMoments(J, X, mmax=MM, kmax=30.0, lams=LAMS)
    two = pm.two_loop(P)
    print(f"two-loop done {time.time() - t0:.0f}s", flush=True)
    NB = pm.norm(P)[0]
    res = pm.moments(P, two=two)
    raw = {'lams': LAMS, 'M': F.M, 'grid': 'nr32 ncheb8; one (64,32,24); two (16,14,10)/(16,14,10,10); kmax 30'}
    for fl, tau in (('u', np.diag([1., 0.])), ('d', np.diag([0., 1.]))):
        o = pm.one_loop(P, tau)
        raw[fl] = {k: (o[k] / NB).real.tolist() for k in ('q_s', 'q_a', 'D_s', 'D_a')}
        raw[fl]['two'] = (res[fl]['4'] + res[fl]['5'] + res[fl]['6']).real.tolist()
    RAW.write_text(json.dumps(raw))
    return raw

def extrap(lams, v, lo, hi, quad=True):
    sel = (lams >= lo) & (lams <= hi)
    l = lams[sel]
    A = [np.ones_like(l), l, l * np.log(l)] + ([l**2] if quad else [])
    c, *_ = np.linalg.lstsq(np.array(A).T, v[sel], rcond=None)
    return c[0]

def analyse(raw):
    from qdq.diquark_df import quark_in_diquark
    lams = np.array(raw['lams'])
    MM = np.array(raw['u']['q_s']).shape[1] - 1                         # as many moments as the raw data have
    qd = quark_in_diquark(mmax=MM)
    z = {'s': qd['0+']['z'], 'a': qd['1+']['z']}
    out = {'M': raw['M'], 'grid': raw['grid']}
    for fl in 'ud':
        r = raw[fl]
        comp = {'1s': np.array(r['q_s']), '1a': np.array(r['q_a']), '2s': np.array(r['D_s']) * z['s'],
                '2a': np.array(r['D_a']) * z['a'], 'two': np.array(r['two'])}
        cen, err = 0, 0
        parts = {}
        for name, v in comp.items():
            A = np.array([extrap(lams, v[:, m], 0.005, 0.05) for m in range(MM + 1)])
            B = np.array([extrap(lams, v[:, m], 0.005, 0.10) for m in range(MM + 1)])
            C = np.array([extrap(lams, v[:, m], 0.005, 0.05, quad=False) for m in range(MM + 1)])
            e = np.maximum(abs(A - B), abs(A - C))
            parts[name] = A.tolist()
            cen, err = cen + A, err + e
        rel = err / abs(cen)
        mono = np.concatenate([[True], np.diff(cen) < 0])
        ok = (rel < 0.05) & np.cumprod(mono).astype(bool)
        mrel = int(np.max(np.nonzero(ok)[0]))
        out[fl] = {'moments': cen.tolist(), 'error': err.tolist(), 'm_reliable': mrel, 'parts': parts}
    return out

if __name__ == '__main__':
    np.set_printoptions(precision=5, suppress=True, linewidth=160)
    raw = json.loads(RAW.read_text()) if '--analyse' in sys.argv else compute()
    out = analyse(raw)
    for fl in 'ud':
        o = out[fl]
        k = o['m_reliable']
        print(f"<x^m>_{fl}, m = 0..{k} (reliable: rel. error < 5 %, monotonic):")
        print("   ", np.array(o['moments'][:k + 1]))
        print("  +-", np.array(o['error'][:k + 1]))
    s1 = out['u']['moments'][1] + out['d']['moments'][1]
    print(f"momentum sum <x>_u + <x>_d = {s1:.5f}")
    json.dump(out, open('results/pdf_moments_highm.json', 'w'), indent=1)
    print("saved results/pdf_moments_highm.json")
