"""Helicity-dependent valence DFs of the proton: Mellin moments <x^m>_{Delta q} at the hadron scale,
AO evolution, comparison with Cheng et al. PLB 844 (2023) 138074 (realistic CSM), lattice and fits.

Axial current at Q = 0 (qdq/pdf_pol.py; Chen et al. 2103.02054 vertices), diquark couplings from the
quark loop as functions of the diquark virtuality (results/axial_couplings_table.json, computed by
AxialCouplingTable.compute).  Numerics as scripts/12: amplitude nr = 32, ncheb = 8; one-loop
(64,32,24); two-loop (16,14,10)/(16,14,10,10); regulator (1 + lam p^2)^-m, lam -> 0 with
{1, lam, lam log lam, lam^2}.  Diagrams 2, 3: <x^m> = <y^m>_D <z^m>, z from phi_1^2 (1+) and
phi_0 phi_1 (transition).

Usage:  python scripts/13_pdf_helicity.py            # full computation (~1 h on 4 cores)
        python scripts/13_pdf_helicity.py --analyse  # re-analyse results/pdf_helicity_raw.json
"""
import sys, json, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

MM = 16
LAMS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
RAW = pathlib.Path('results/pdf_helicity_raw.json')

def compute():
    from qdq.faddeev import NucleonFaddeev
    from qdq.current import NucleonCurrent
    from qdq.exchange import ExchangeCurrent
    from qdq.pdf_pol import NucleonHelicityMoments, AxialCouplingTable
    t0 = time.time()
    F = NucleonFaddeev(g=(14.94 / 2, 1j * 12.98 / 2), nr=32, nz=10, ny=8, ncheb=8)
    F.solve_mass(1.17, 1.19, verbose=False)
    P = np.array([0, 0, 0, 1j * F.M])
    J = NucleonCurrent(F, nk=64, nz=32, ny=24)
    X = ExchangeCurrent(J, nk=(16, 14, 10), nl=(16, 14, 10, 10))
    tab = AxialCouplingTable.load()
    hm = NucleonHelicityMoments(J, X, tab, mmax=MM, kmax=30.0, lams=LAMS)
    two = hm.two_loop(P)
    print(f"two-loop done {time.time() - t0:.0f}s", flush=True)
    r = hm.helicity_moments(P, two=two)
    raw = {'lams': LAMS, 'M': F.M, 'grid': 'nr32 ncheb8; one (64,32,24); two (16,14,10)/(16,14,10,10); kmax 30',
           'phase': [r['phase'].real, r['phase'].imag]}
    for fl in 'ud':
        raw[fl] = {k: np.real(r[fl][k]).tolist() for k in ('1s', '1a', '4')}
        raw[fl].update({k: np.real(v).tolist() for k, v in r[fl]['diquark_y'].items()})
        raw[fl]['max_imag'] = float(np.abs(np.imag(r[fl]['total'])).max())
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
    MM = np.array(raw['u']['1s']).shape[1] - 1                         # as many moments as the raw data have
    qd = quark_in_diquark(mmax=MM)
    out = {'M': raw['M'], 'grid': raw['grid']}
    for fl in 'ud':
        r = raw[fl]
        comp = {'1s': np.array(r['1s']), '1a': np.array(r['1a']), '2': np.array(r['D_aa']) * qd['1+']['z'],
                '3': (np.array(r['D_sa']) + np.array(r['D_as'])) * qd['01']['z'], '4': np.array(r['4'])}
        cen, err, parts = 0, 0, {}
        for name, v in comp.items():
            A = np.array([extrap(lams, v[:, m], 0.005, 0.05) for m in range(MM + 1)])
            B = np.array([extrap(lams, v[:, m], 0.005, 0.10) for m in range(MM + 1)])
            C = np.array([extrap(lams, v[:, m], 0.005, 0.05, quad=False) for m in range(MM + 1)])
            parts[name] = A.tolist()
            cen, err = cen + A, err + np.maximum(abs(A - B), abs(A - C))
        rel = err / abs(cen)
        mono = np.concatenate([[True], np.diff(abs(cen)) < 0])
        ok = (rel < 0.05) & np.cumprod(mono).astype(bool)
        out[fl] = {'moments': cen.tolist(), 'error': err.tolist(), 'm_reliable': int(np.max(np.nonzero(ok)[0])),
                   'parts': parts}
    return out

if __name__ == '__main__':
    np.set_printoptions(precision=5, suppress=True, linewidth=160)
    raw = json.loads(RAW.read_text()) if '--analyse' in sys.argv else compute()
    out = analyse(raw)
    for fl in 'ud':
        o = out[fl]; k = o['m_reliable']
        print(f"<x^m>_Delta{fl}, m = 0..{k}:", np.array(o['moments'][:k + 1]), "\n   +-", np.array(o['error'][:k + 1]))
        print("   diagram parts at m = 0:", {p: round(v[0], 4) for p, v in o['parts'].items()})
    gA = out['u']['moments'][0] - out['d']['moments'][0]
    aI = out['u']['moments'][0] + out['d']['moments'][0]
    print(f"g_A = {gA:.4f};  Delta u/g_A = {out['u']['moments'][0] / gA:.3f}, Delta d/g_A = {out['d']['moments'][0] / gA:.3f}, "
          f"a_I/g_A = {aI / gA:.3f}   [Cheng et al. 2023: 0.762, -0.238(6), 0.52(1); g_A = 1.25]")
    # ---- AO evolution of the valence (non-singlet) helicity moments: LO Delta P_qq = P_qq --------------
    from qdq.evolution import AOEvolution
    ev = AOEvolution()
    k = min(out['u']['m_reliable'], out['d']['m_reliable'], 3)
    H = {fl: np.array(out[fl]['moments'][:k + 1]) for fl in 'ud'}
    for z in (2.0,):
        fac = np.array([1.0] + [ev.valence_factor(n, z) for n in range(1, k + 1)])
        Hz = {fl: H[fl] * fac for fl in 'ud'}
        out[f'valence_{z}'] = {fl: Hz[fl].tolist() for fl in 'ud'}
        iv = Hz['u'] - Hz['d']
        print(f"zeta = {z} GeV: <x^m>_Du = {Hz['u']}, <x^m>_Dd = {Hz['d']}")
        print(f"  isovector <x>_(Du-Dd) = {iv[1]:.4f}   [PNDME lattice 0.213(15)(22); NNPDFpol1.1 0.195(14), "
              f"DSSV08 0.203(9), JAM17 0.241(26)]")
        if k >= 2:
            print(f"  isovector <x^2>_(Du-Dd) = {iv[2]:.4f}   [Taggi et al. lattice 0.094(15); NNPDFpol1.1 0.0659(46), "
                  f"NNPDFpol2.0 0.0586(46), MAPPDFpol1.0 0.0666(53); valence only here]")
    json.dump(out, open('results/pdf_helicity_moments.json', 'w'), indent=1)
    print("saved results/pdf_helicity_moments.json")
