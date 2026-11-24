"""Hadron-scale valence DFs of Lambda and Sigma^0 (and the proton in the same setup): Mellin moments
<x^m>_q, q = u, d, s, m = 0..6, from the flavour-resolved quark+diquark Faddeev amplitude
(qdq/baryon_faddeev.py, qdq/baryon_pdf.py).  Setup A of scripts/17: diquark masses 0.79, 0.89 GeV
(2611.19819) + 0.15 GeV per s quark, all couplings from the canonical normalisation, strange quark of
1901.04305 App. A.

Steps:
  1. validation: the proton from BaryonFaddeev with the couplings of scripts/09 reproduces
     NucleonPDFMoments (results/pdf_moments_step1.json, kmax = 15);
  2. quark-in-diquark DFs q(z) ~ phi^2 for [ud], {ud} (light) and [us], {us} (asymmetric DA, fitted to
     <xi^1..6>; light quark ~ phi(z)^2, strange ~ phi(1-z)^2);
  3. per baryon: mass, normalisation, diagram breakdown; checks u, d, s numbers and the momentum sum;
     UV-cutoff stability (kmax 10 vs 15 GeV).
Grids as scripts/09 (Faddeev nr = 16, nz = ny = 8, ncheb = 6; one loop 48 x 32 x 24; two loop
(12,6,6) x (12,6,6,6)): moments reliable to m ~ 6.

Usage: python scripts/18_hyperon_pdf.py            -> results/hyperon_pdf_moments.json
       python scripts/18_hyperon_pdf.py --su3      -> SU(3)-limit check of all flavour factors
                                                     (results/hyperon_pdf_su3_check.json)
       python scripts/18_hyperon_pdf.py --analyse  -> AO evolution to 2 GeV, comparison with the SCI
                                                     results of arXiv:2505.10663 (Eq. 59, Table 4)
       python scripts/18_hyperon_pdf.py --scan     -> sensitivity of the SU(3) breaking to the inputs
                                                     (results/hyperon_pdf_scan.json)
"""
import sys, json, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qdq.baryon_faddeev import BaryonFaddeev, QUARKS, diquark_coupling
from qdq.baryon_pdf import BaryonPDFMoments
from qdq.diquark_df import da_moments_flavour, fit_da_gen, _xi_moments_from_x

np.set_printoptions(precision=5, suppress=True, linewidth=160)
MM = 6
SETUP_A = {('s', 'll'): 0.79, ('a', 'll'): 0.89, ('s', 'ls'): 0.94, ('a', 'ls'): 1.04, ('a', 'ss'): 1.19}
GRID = dict(nr=16, nz=8, ny=8, ncheb=6)
STARTS = {'p': (1.16, 1.19), 'Lambda': (1.34, 1.38), 'Sigma0': (1.37, 1.40)}


def couplings(ms, quarks=QUARKS):
    g = {}
    for (J, dq), m in ms.items():
        qa = quarks['l'] if dq[0] == 'l' else quarks['s']
        qb = quarks['l'] if dq[1] == 'l' else quarks['s']
        g[J, dq] = diquark_coupling(J, m, qa, qb)[0]
    return g


def zmoments(ms, mmax=MM, kmax=20.0, quarks=QUARKS):
    """<z^m> of the struck quark for (J, dq, struck class) + DA info."""
    z, info = {}, {}
    for (J, dq), m in ms.items():
        if dq == 'ss':
            continue
        qa = quarks['l']
        qb = quarks['l'] if dq == 'll' else quarks['s']
        xm, _ = da_moments_flavour(J, m, qa, qb, mmax=8, kmax=kmax)
        d, xim, cost = fit_da_gen(xm.real)
        z[J, dq, 'l'] = d.square_moments(mmax)
        if dq == 'ls':
            z[J, dq, 's'] = d.square_moments(mmax, reflect=True)
        info[f'{J}{dq}'] = dict(mass=m, xi_moments=xim[1:9].tolist(), alpha=d.alpha, a=d.a.tolist(),
                                fit_cost=cost, phi_x_light=xm[1].real,
                                z_light=z[J, dq, 'l'].tolist(),
                                z_strange=z[J, dq, 's'].tolist() if dq == 'ls' else None)
        print(f"  DA {J}{dq} (m = {m}): <xi^1..6> = {xim[1:7]}, alpha = {d.alpha:.4f}, a = {d.a}")
        print(f"      <z^m> light   = {z[J, dq, 'l']}")
        if dq == 'ls':
            print(f"      <z^m> strange = {z[J, dq, 's']}")
    return z, info


def tolist(r):
    out = {}
    for q in 'uds':
        out[q] = {k: (np.real(v).tolist() if not isinstance(v, dict) else {c: np.real(w).tolist() for c, w in v.items()})
                  for k, v in r[q].items()}
    nb = r['norm']
    out['norm'] = dict(NB=complex(nb['NB']).real, dlam=complex(nb['dlam']).real, n0=complex(nb['n0']).real,
                       n0_groups=[complex(v).real for v in nb['n0_groups']])
    return out


def su3_check():
    """SU(3) limit (m_s = m_l, degenerate diquarks, equal couplings): octet relations moment by moment,
    Sigma0: u = d = u_p/2, s = d_p;  Lambda: u = d = (u_p + 4 d_p)/6, s = (2 u_p - d_p)/3."""
    t0 = time.time()
    su3 = dict(quarks={'s': QUARKS['l']}, masses={('s', 'ls'): 0.79, ('a', 'ls'): 0.89, ('a', 'ss'): 0.89},
               couplings={('s', 'll'): 7.47, ('a', 'll'): 6.49j, ('s', 'ls'): 7.47, ('a', 'ls'): 6.49j,
                          ('a', 'ss'): 6.49j})
    z, _ = zmoments({('s', 'll'): 0.79, ('a', 'll'): 0.89})
    for J in 'sa':
        z[J, 'ls', 'l'] = z[J, 'ls', 's'] = z[J, 'll', 'l']
    res = {}
    for b in ('p', 'Lambda', 'Sigma0'):
        F = BaryonFaddeev(b, **su3, **GRID)
        F.solve_mass(1.17, 1.19)
        r = BaryonPDFMoments(F, mmax=MM).moments(np.array([0, 0, 0, 1j * F.M]), z_moments=z)
        res[b] = {q: {k: np.real(r[q][k]) for k in ('total', '1', '2', '4', '5', '6')} for q in 'uds'}
        print(f"  {b:7s} M = {F.M:.8f}   ({time.time()-t0:.0f}s)")
    p = res['p']
    pred = {'Sigma0': {'u': lambda k: p['u'][k] / 2, 's': lambda k: p['d'][k]},
            'Lambda': {'u': lambda k: (p['u'][k] + 4 * p['d'][k]) / 6, 's': lambda k: (2 * p['u'][k] - p['d'][k]) / 3}}
    out = {}
    for b in ('Lambda', 'Sigma0'):
        for q in 'us':
            dev = max(np.abs(res[b][q][k] - pred[b][q](k)).max() for k in ('total', '1', '2', '4', '5', '6'))
            out[f'{b}_{q}'] = dev
            print(f"  SU(3): {b:7s} {q}: <x^m> = {res[b][q]['total']}, max |dev| (all diagrams) = {dev:.1e}")
    return out


# arXiv:2505.10663 (SCI): Eq. (59) hadron-scale momentum fractions; Table 4 zeta_2 = 2 GeV moments (%)
SCI_H = {'p': {'u': 2 * 0.343, 'd': 0.314, 's': 0.0}, 'Lambda': {'u': 0.303, 'd': 0.303, 's': 0.394},
         'Sigma0': {'u': 0.328, 'd': 0.328, 's': 0.345}}
SCI_2 = {'p': {'u': [32.7, 9.47, 3.58], 'd': [15.0, 4.15, 1.54]},
         'Lambda': {'u+d': [28.9, 7.60, 2.66], 's': [18.8, 5.83, 2.29]},
         'Sigma0': {'u+d': [31.3, 8.46, 2.99], 's': [16.4, 4.70, 1.74]}}


def analyse(path='results/hyperon_pdf_moments.json'):
    from qdq.evolution import AOEvolution, ZETA_H
    r = json.load(open(path))
    ev = AOEvolution()
    fac = np.array([1.0] + [ev.valence_factor(n, 2.0) for n in range(1, MM + 1)])
    out = {'zeta_H': ZETA_H, 'valence_factor_2GeV': fac.tolist()}
    print(f"zeta_H = {ZETA_H} GeV; valence factors to 2 GeV (n = 1..3): {fac[1:4]}")
    print(f"\n{'':8s} {'q':4s} {'<x>(zeta_H)':>12s} {'SCI':>7s} | {'<x^n>(2 GeV) %, n=1,2,3':>28s} | {'SCI Table 4':>20s}")
    for b in ('p', 'Lambda', 'Sigma0'):
        mom = {q: np.array(r[b]['kmax15'][q]['total']) for q in 'uds'}
        out[b] = {'hadron': {q: mom[q].tolist() for q in 'uds'}, '2GeV': {q: (mom[q] * fac).tolist() for q in 'uds'}}
        rows = [('u', mom['u']), ('d', mom['d']), ('s', mom['s'])] if b == 'p' else \
               [('u+d', mom['u'] + mom['d']), ('s', mom['s'])]
        for q, v in rows:
            sci_h = SCI_H[b]['u'] + SCI_H[b]['d'] if q == 'u+d' else SCI_H[b][q]
            e2 = 100 * v[1:4] * fac[1:4]
            sci2 = SCI_2[b].get(q)
            print(f"{b:8s} {q:4s} {v[1]:12.4f} {sci_h:7.3f} | {e2[0]:8.2f} {e2[1]:8.2f} {e2[2]:8.3f}   |"
                  f" {'' if sci2 is None else sci2}")
        # ratios of higher moments: a proxy for the large-x ratio s/l
        if b != 'p':
            print(f"{'':8s} <x^m>_s/<x^m>_u, m = 1..6: {np.round(mom['s'][1:] / mom['u'][1:], 3)}")
        else:
            print(f"{'':8s} <x^m>_d/<x^m>_u, m = 1..6: {np.round(mom['d'][1:] / mom['u'][1:], 3)}")
        out[b]['diagrams'] = {q: {k: r[b]['kmax15'][q][k][:2] for k in '12456'} for q in 'uds'}
        dk = max(abs(np.array(r[b]['kmax15'][q]['total'][m]) - np.array(r[b]['kmax10'][q]['total'][m]))
                 for q in 'uds' for m in range(MM + 1))
        print(f"{'':8s} kmax 10 vs 15 GeV: max |diff| = {dk:.1e}")
    json.dump(out, open('results/hyperon_pdf_evolved.json', 'w'), indent=1)
    print("saved results/hyperon_pdf_evolved.json")


def spaced(d):
    """2611.19819 light diquarks + d GeV per s quark."""
    return {('s', 'll'): 0.79, ('a', 'll'): 0.89, ('s', 'ls'): 0.79 + d, ('a', 'ls'): 0.89 + d, ('a', 'ss'): 0.89 + 2 * d}

SCAN = {
    'A': dict(masses=SETUP_A),
    'A10': dict(masses=spaced(0.10)),
    'A20': dict(masses=spaced(0.20)),
    'C': dict(masses={('s', 'll'): 0.80, ('a', 'll'): 0.90, ('s', 'ls'): 0.95, ('a', 'ls'): 1.05, ('a', 'ss'): 1.20}),
    'A_s15': dict(masses=SETUP_A, s_mbar=1.5),
}
LATTICE_SIGMA = {'u_Sig/u_p': 0.9631, 's_Sig/d_p': 1.082}     # 1012.0215 Table I, m_pi = 334, m_K = 463 MeV


def scan(path='results/hyperon_pdf_scan.json'):
    """Sensitivity of the SU(3) breaking to the inputs: diquark mass spacing per s quark (0.10, 0.15,
    0.20 GeV), setup C (1901.04305 masses), and a heavier s quark (mbar_s x 1.5: M^E_s 0.495 -> 0.552 GeV).
    Ratios that are scale independent under valence evolution: Sigma+ (= isospin rotation of Sigma0:
    u = 2 u_Sigma0, s = s_Sigma0) vs proton, and s/u in Lambda."""
    t0 = time.time()
    out = json.load(open(path)) if pathlib.Path(path).exists() else {}
    base = json.load(open('results/hyperon_pdf_moments.json'))
    for name, cfg in SCAN.items():
        if name in out:
            continue
        ms = cfg['masses']
        quarks = dict(QUARKS)
        if 's_mbar' in cfg:
            quarks['s'] = dict(QUARKS['s'], mbar=QUARKS['s']['mbar'] * cfg['s_mbar'])
        res = {'masses': {f'{J}{dq}': m for (J, dq), m in ms.items()}, 's_mbar_factor': cfg.get('s_mbar', 1.0)}
        if name == 'A':
            for b in ('p', 'Lambda', 'Sigma0'):
                res[b] = {'M': base[b]['M'], **{q: base[b]['kmax15'][q]['total'] for q in 'uds'}}
        else:
            print(f"setup {name}: masses {res['masses']}, mbar_s x {res['s_mbar_factor']}", flush=True)
            g = couplings(ms, quarks)
            z, _ = zmoments(ms, quarks=quarks)
            for b, (m0, m1) in STARTS.items():
                if b == 'p' and ms[('s', 'll')] == 0.79 and ms[('a', 'll')] == 0.89:
                    res[b] = {'M': base[b]['M'], **{q: base[b]['kmax15'][q]['total'] for q in 'uds'}}
                    continue
                F = BaryonFaddeev(b, quarks=quarks, masses=ms, couplings=g, **GRID)
                M = F.solve_mass(m0 - 0.05, m1 + 0.05)
                r = BaryonPDFMoments(F, mmax=MM).moments(np.array([0, 0, 0, 1j * M]), z_moments=z)
                res[b] = {'M': M, **{q: np.real(r[q]['total']).tolist() for q in 'uds'}}
                print(f"   {b:7s} M = {M:.4f}  <x>_u,s = {res[b]['u'][1]:.4f}, {res[b]['s'][1]:.4f}  "
                      f"numbers {res[b]['u'][0]:.4f}, {res[b]['s'][0]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        p, L, S = res['p'], res['Lambda'], res['Sigma0']
        res['ratios'] = {'u_Sig/u_p': 2 * S['u'][1] / p['u'][1], 's_Sig/d_p': S['s'][1] / p['d'][1],
                         's/u_Lambda': L['s'][1] / L['u'][1], 's/u_Sigma0': S['s'][1] / S['u'][1]}
        out[name] = res
        json.dump(out, open(path, 'w'), indent=1)
    print(f"
{'setup':7s} {'M_p':>6s} {'M_L':>6s} {'M_S':>6s} {'<x>_u^L':>8s} {'<x>_s^L':>8s} {'<x>_u^S':>8s} "
          f"{'<x>_s^S':>8s} {'s/u L':>6s} {'uS/up':>6s} {'sS/dp':>6s}")
    for name, r in out.items():
        q = r['ratios']
        print(f"{name:7s} {r['p']['M']:6.3f} {r['Lambda']['M']:6.3f} {r['Sigma0']['M']:6.3f} {r['Lambda']['u'][1]:8.4f} "
              f"{r['Lambda']['s'][1]:8.4f} {r['Sigma0']['u'][1]:8.4f} {r['Sigma0']['s'][1]:8.4f} {q['s/u_Lambda']:6.3f} "
              f"{q['u_Sig/u_p']:6.3f} {q['s_Sig/d_p']:6.3f}")
    print(f"lattice (1012.0215, m_pi = 334, m_K = 463 MeV): uS/up = 0.963(9), sS/dp = 1.082(18); "
          f"quenched Lambda (hep-lat/0208017): s/u = 1.35(8)")


def main():
    if '--scan' in sys.argv:
        scan()
        return
    if '--analyse' in sys.argv:
        analyse()
        return
    if '--su3' in sys.argv:
        out = su3_check()
        pathlib.Path('results').mkdir(exist_ok=True)
        json.dump(out, open('results/hyperon_pdf_su3_check.json', 'w'), indent=1)
        return
    t0 = time.time()
    out = {'setup': {f'{J}{dq}': m for (J, dq), m in SETUP_A.items()}, 'grid': GRID}
    # ---- 1. validation against NucleonPDFMoments
    ref = json.load(open('results/pdf_moments_step1.json'))['kmax15']
    F = BaryonFaddeev('p', couplings={('s', 'll'): 7.47, ('a', 'll'): 6.49j}, **GRID)
    F.solve_mass(1.17, 1.19)
    r = BaryonPDFMoments(F, mmax=MM).moments(np.array([0, 0, 0, 1j * F.M]))
    dev = max(abs(r[q]['total'][m].real - ref[q]['total'][m]) for q in 'ud' for m in range(MM + 1))
    print(f"validation (proton, couplings of scripts/09): M = {F.M:.6f}, max |diff| vs NucleonPDFMoments = {dev:.1e}")
    out['validation_maxdiff'] = dev
    # ---- 2. quark-in-diquark DFs
    print("quark-in-diquark DFs (DA^2 prescription):")
    z, info = zmoments(SETUP_A)
    out['diquark_da'] = info
    # ---- 3. baryons
    g = couplings(SETUP_A)
    out['couplings'] = {f'{J}{dq}': [complex(v).real, complex(v).imag] for (J, dq), v in g.items()}
    for b, (m0, m1) in STARTS.items():
        F = BaryonFaddeev(b, masses=SETUP_A, couplings=g, **GRID)
        M = F.solve_mass(m0, m1)
        P = np.array([0, 0, 0, 1j * M])
        print(f"\n{b}: M = {M:.5f} GeV  groups {[gr['name'] for gr in F.groups]}   ({time.time()-t0:.0f}s)")
        out[b] = {'M': M, 'groups': [gr['name'] for gr in F.groups]}
        for kmax in (15.0, 10.0):
            pm = BaryonPDFMoments(F, mmax=MM, kmax=kmax)
            nb = pm.norm(P)
            one = pm.one_loop(P)
            two = pm.two_loop(P)
            r = pm.moments(P, z_moments=z, one=one, two=two, nb=nb)
            tag = f'kmax{int(kmax)}'
            out[b][tag] = tolist(r)
            tot = sum(r[q]['total'] for q in 'uds')
            print(f"  kmax = {kmax:4.0f}: n0 per group = {np.round(np.real(nb[3]) / np.real(nb[2]), 4)}  ({time.time()-t0:.0f}s)")
            for q in 'uds':
                print(f"    <x^m>_{q} = {r[q]['total'].real}   max|Im| {np.abs(r[q]['total'].imag).max():.0e}")
                if kmax == 15.0:
                    for k in '12456':
                        print(f"        diagram {k}: {np.real(r[q][k])[:4]}")
            print(f"    sum_q: number {tot[0].real:.5f}, momentum {tot[1].real:.5f}")
    pathlib.Path('results').mkdir(exist_ok=True)
    json.dump(out, open('results/hyperon_pdf_moments.json', 'w'), indent=1)
    print(f"saved results/hyperon_pdf_moments.json  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
