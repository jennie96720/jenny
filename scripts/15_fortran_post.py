"""Post-processing of the Fortran DF runs (fortran/, modes 'pdf' and 'helicity'):
reads fortran/outdata/<root>/{pdf,helicity}_raw.txt, builds the same raw data as scripts/12 and 13,
and applies their analysis (lam -> 0 extrapolation with {1, lam, lam log lam, lam^2}, quark-in-diquark DFs
phi^2 / phi_0 phi_1, reliability cut) and the AO evolution of scripts/11, 13.  With fewer than five
lam > 0 values (quick runs) the lam = 0 moments are used directly (reliable to m ~ 6).

Usage:  python scripts/15_fortran_post.py [fortran/outdata/pdf]
"""
import sys, json, pathlib, importlib.util; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'fortran/outdata/pdf')

def load(path):
    head, rows = {}, {}
    for line in open(path):
        if line.startswith('#'):
            if '=' in line:
                k, v = line[1:].split('=', 1)
                head[k.strip()] = v.strip()
            continue
        fl, c, l, m, re, im = line.split()
        rows.setdefault((fl, c), {})[(int(l) - 1, int(m))] = complex(float(re), float(im))
    lams = [float(x) for x in head['lams'].split()]
    nlam, mmax = len(lams), max(m for d in rows.values() for (_, m) in d)
    arr = {k: np.array([[d[(l, m)] for m in range(mmax + 1)] for l in range(nlam)]) for k, d in rows.items()}
    return head, lams, arr

def module(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

np.set_printoptions(precision=5, suppress=True, linewidth=160)
from qdq.diquark_df import quark_in_diquark
from qdq.evolution import AOEvolution
ev = AOEvolution()
out = {}
for kind in ('pdf', 'helicity'):
    path = ROOT / f'{kind}_raw.txt'
    if not path.exists():
        continue
    head, lams, A = load(path)
    print(f"== {path}  (M = {head['M']}, lams = {lams})")
    if kind == 'pdf':
        raw = {'lams': lams, 'M': float(head['M']), 'grid': 'fortran'}
        for fl in 'ud':
            raw[fl] = {k: A[(fl, k)].real.tolist() for k in ('q_s', 'q_a', 'D_s', 'D_a')}
            raw[fl]['two'] = sum(A[(fl, c)] for c in ('4', '5', '6')).real.tolist()
        ana = module('scripts/12_pdf_high_moments.py').analyse
    else:
        raw = {'lams': lams, 'M': float(head['M']), 'grid': 'fortran'}
        for fl in 'ud':
            raw[fl] = {k: A[(fl, k)].real.tolist() for k in ('1s', '1a', '4', 'D_aa', 'D_sa', 'D_as')}
        ana = module('scripts/13_pdf_helicity.py').analyse
    if sum(l > 0 for l in lams) >= 5:
        res = ana(raw)
        mom = {fl: np.array(res[fl]['moments'][:res[fl]['m_reliable'] + 1]) for fl in 'ud'}
    else:                                                    # quick run: lam = 0 values, m <= 6
        mm = min(len(raw['u'][next(iter(raw['u']))][0]) - 1, 6)
        qd = quark_in_diquark(mmax=mm)
        i0 = lams.index(0.0)
        mom = {}
        for fl in 'ud':
            r = {k: np.array(v)[i0, :mm + 1] for k, v in raw[fl].items()}
            if kind == 'pdf':
                mom[fl] = r['q_s'] + r['q_a'] + r['D_s'] * qd['0+']['z'] + r['D_a'] * qd['1+']['z'] + r['two']
            else:
                mom[fl] = r['1s'] + r['1a'] + r['D_aa'] * qd['1+']['z'] + (r['D_sa'] + r['D_as']) * qd['01']['z'] + r['4']
    lab = '' if kind == 'pdf' else 'Delta'
    for fl in 'ud':
        print(f"  <x^m>_{lab}{fl}(zeta_H) =", mom[fl])
    k = min(len(mom['u']), len(mom['d']), 4) - 1
    fac = np.array([1.0] + [ev.valence_factor(n, 3.097) for n in range(1, k + 1)])
    if kind == 'pdf':
        print(f"  zeta_3 = 3.097 GeV valence (%): u {100 * mom['u'][1:k + 1] * fac[1:]}, d {100 * mom['d'][1:k + 1] * fac[1:]}"
              "   [realistic CSM: u 30.4, 7.7, 2.5; d 13.8, 3.2, 0.9]")
    else:
        gA = mom['u'][0] - mom['d'][0]
        f2 = np.array([1.0] + [ev.valence_factor(n, 2.0) for n in range(1, k + 1)])
        print(f"  g_A = {gA:.4f}, Delta u/g_A = {mom['u'][0] / gA:.3f}, Delta d/g_A = {mom['d'][0] / gA:.3f};"
              f"  <x>_(Du-Dd)(2 GeV) = {(mom['u'][1] - mom['d'][1]) * f2[1]:.4f}")
    out[kind] = {fl: mom[fl].tolist() for fl in 'ud'}
json.dump(out, open(ROOT / 'post.json', 'w'), indent=1)
print(f"saved {ROOT / 'post.json'}")
