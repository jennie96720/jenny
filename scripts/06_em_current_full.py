"""Milestone 5c: full nucleon EM current (diagrams 1-6) and form factors.

Checks at Q = 0:
  a) two-loop machinery:   Psibar G0 K G0 Psi == Psibar G0 Psi      (Faddeev eigenvalue = 1)
  b) Ward identity, baryon charges: diagram 4 == -Psibar G0 dK/dP G0 Psi (eta = 1/3: only the
     exchanged quark depends on P), seagulls 5,6 == 0
  c) charges: baryon number = 1,  F1p(0) = 1,  F1n(0) = 0   (normalisation from dlambda/dM only)
Then G_E, G_M of p and n vs x = Q^2/M^2, magnetic moments and radii, compared with
Cheng et al. arXiv:2507.13484 Table 2.

Usage: python3 scripts/06_em_current_full.py [coarse|fine]
"""
import sys, time, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qdq.faddeev import NucleonFaddeev, flavour_coeffs
from qdq.current import NucleonCurrent, EQ, EB, HBARC
from qdq.exchange import ExchangeCurrent, exchange_flavour

level = sys.argv[1] if len(sys.argv) > 1 else 'coarse'
grids = {'coarse': ((12, 6, 6), (12, 6, 6, 6)), 'fine': ((16, 8, 8), (16, 8, 8, 8))}
nk, nl = grids[level]
PHI = {'p': np.array([1., 0.]), 'n': np.array([0., 1.])}

F = NucleonFaddeev(g=(14.94 / 2, 1j * 12.98 / 2), nr=16, nz=8, ny=8, ncheb=6)
F.solve_mass(1.17, 1.19, verbose=False)
M = F.M
J = NucleonCurrent(F, nk=32, nz=16, ny=12)
B, F1G0, dlam, n0 = J.normalisation()
X = ExchangeCurrent(J, nk=nk, nl=nl)
print(f"M_N = {M:.4f} GeV, two-loop grid outer {nk} inner {nl}")

print("a) two-loop identity")
c = flavour_coeffs()
r0 = X.compute(0.0, photon=False)
K = sum(c[XY] * r0[XY]['K'][0] for XY in r0) / 2
print(f"   Psibar G0 K G0 Psi = {K.real:.7f},  Psibar G0 Psi = {n0.real:.7f},  ratio {(K / n0).real:.5f}")

print("b) Ward identity at Q = 0 (baryon charges)")
rd = X.compute(0.0, deriv=True)
dK = sum(c[XY] * rd[XY]['dK'][0] for XY in rd)
rq0 = X.compute(0.0)
tb = ExchangeCurrent.combine(rq0, exchange_flavour(EB, PHI['p']))
print(f"   diagram 4 = {tb['4'][0].imag:.7f} i,  -Psibar G0 dK/dP G0 Psi = {dK.imag:.7f} i,"
      f"  diagrams 5, 6 = {abs(tb['5'][0]):.1e}, {abs(tb['6'][0]):.1e}")

print("c) charges (normalised by the eigenvalue formula, no fit)")
F10 = {}
d12 = J.diagrams(0.0, charge=EB, which=(1, 2))
bn = (d12[1][0] + d12[2][0] + tb['4'][0] + tb['5'][0] + tb['6'][0]) / B
print(f"   baryon number = {bn.real:.5f}")
for nm, phi in PHI.items():
    te = ExchangeCurrent.combine(rq0, exchange_flavour(EQ, phi))
    d = J.diagrams(0.0, charge=EQ, phi=phi)
    parts = [d[1][0], d[2][0], d[3][0], te['4'][0], te['5'][0], te['6'][0]]
    F10[nm] = (sum(parts) / B).real
    print(f"   F1{nm}(0) = {(sum(parts) / B).real:+.5f}   [diagrams 1..6: "
          + " ".join(f"{(v / B).real:+.4f}" for v in parts) + "]")

print("d) form factors")
xs = [0.005, 0.02, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
out = {'M': M, 'grid': [nk, nl], 'x': xs, 'p': [], 'n': []}
print("    x      | GEp      GMp      | GEn      GMn      | F1p     F2p     F1n     F2n")
for x in xs:
    Q = np.sqrt(x) * M
    t = time.time()
    rq = X.compute(Q)
    row = {}
    for nm, phi in PHI.items():
        te = ExchangeCurrent.combine(rq, exchange_flavour(EQ, phi))
        d = J.diagrams(Q, charge=EQ, phi=phi)
        tot = (sum(d.values()) + te['4'] + te['5'] + te['6']) / B
        F1, F2 = tot[0].real, tot[1].real
        row[nm] = dict(F1=F1, F2=F2, GE=F1 - x / 4 * F2, GM=F1 + F2, F3=abs(tot[2]),
                       diag={str(i): (d[i] / B).real[:2].tolist() for i in (1, 2, 3)}
                       | {k: (te[k] / B).real[:2].tolist() for k in ('4', '5', '6')})
        out[nm].append(row[nm])
    p, n = row['p'], row['n']
    print(f"   {x:5.3f}   | {p['GE']:+.4f} {p['GM']:+.4f} | {n['GE']:+.4f} {n['GM']:+.4f} | "
          f"{p['F1']:+.4f} {p['F2']:+.4f} {n['F1']:+.4f} {n['F2']:+.4f}   ({time.time() - t:.0f}s, F3 < {max(p['F3'], n['F3']):.0e})")

# static properties from the two smallest x (quadratic through the G_E(0) computed on the same grid,
# so that the small two-loop discretisation error in the charge cancels in the slope)
def slope(nm, key, val0):
    x1, x2 = xs[0], xs[1]
    g1, g2 = out[nm][0][key], out[nm][1][key]
    # G = val0 + a x + b x^2
    a = ((g1 - val0) / x1 * x2 - (g2 - val0) / x2 * x1) / (x2 - x1)
    return a / M**2                                       # dG/dQ^2
print("e) static properties (this work | 2507.13484 Table 2)")
GMp0 = out['p'][0]['GM'] - (out['p'][1]['GM'] - out['p'][0]['GM']) / (xs[1] - xs[0]) * xs[0]
GMn0 = out['n'][0]['GM'] - (out['n'][1]['GM'] - out['n'][0]['GM']) / (xs[1] - xs[0]) * xs[0]
rEp2 = -6 * slope('p', 'GE', F10['p']) / F10['p'] * HBARC**2
rEn2 = -6 * slope('n', 'GE', F10['n']) * HBARC**2
rMp2 = -6 * slope('p', 'GM', GMp0) / GMp0 * HBARC**2
rMn2 = -6 * slope('n', 'GM', GMn0) / GMn0 * HBARC**2
sq = lambda r2: np.sign(r2) * np.sqrt(abs(r2))
print(f"   mu_p = {GMp0:+.3f} | +2.80      mu_n = {GMn0:+.3f} | -1.86")
print(f"   r_E^p = {sq(rEp2):+.3f} fm | 0.67      r_E^n^2 = {rEn2:+.4f} fm^2 | -0.084 (= -0.29^2)")
print(f"   r_M^p = {sq(rMp2):+.3f} fm | 0.67      r_M^n = {sq(rMn2):+.3f} fm | 0.72")
out['F1_0'] = F10
out['static'] = dict(mu_p=GMp0, mu_n=GMn0, rEp2=rEp2, rEn2=rEn2, rMp2=rMp2, rMn2=rMn2)
pathlib.Path('results').mkdir(exist_ok=True)
json.dump(out, open(f'results/em_form_factors_{level}.json', 'w'), indent=1)
print(f"saved results/em_form_factors_{level}.json")
