"""Milestone 6: nucleon gravitational form factors A, J, D, S, cbar (and B = 2J - A) at the
hadron scale, following arXiv:2611.19819 with our own Faddeev amplitude and current.

Same six diagrams as the EM current (qdq/current.py, qdq/exchange.py) with the graviton probe of
qdq/probes.py.  Checks: total cbar(Q^2) = 0 (EMT conservation), |A(0)| = 1 (momentum sum rule),
J(0) = 1/2 (spin), B(0) = 0.  Breakdowns: diquark content s / a / m, flavour u / d.

Usage: python3 scripts/07_gravitational_ff.py [coarse|fine]
"""
import sys, time, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qdq.faddeev import NucleonFaddeev
from qdq.current import NucleonCurrent
from qdq.exchange import ExchangeCurrent, exchange_flavour
from qdq.probes import GravityProbe, GFF_NAMES

level = sys.argv[1] if len(sys.argv) > 1 else 'coarse'
nk, nl = {'coarse': ((12, 6, 6), (12, 6, 6, 6)), 'fine': ((16, 8, 8), (16, 8, 8, 8))}[level]
PHI = np.array([1., 0.])                                  # proton
TAU = {'tot': np.eye(2), 'u': np.diag([1., 0.]), 'd': np.diag([0., 1.])}

F = NucleonFaddeev(g=(14.94 / 2, 1j * 12.98 / 2), nr=16, nz=8, ny=8, ncheb=6)
F.solve_mass(1.17, 1.19, verbose=False)
M = F.M
J = NucleonCurrent(F, nk=32, nz=16, ny=12)
B = J.normalisation()[0]
gp = GravityProbe((F.ms, F.ma), M)
X = ExchangeCurrent(J, nk=nk, nl=nl)
print(f"M_N = {M:.4f} GeV, two-loop grid outer {nk} inner {nl}")

def evaluate(Q):
    """GFFs for total / u / d and the diquark breakdown s / a / m of the total."""
    raw = X.compute(Q, probe=gp)
    res = {}
    for fl, tau in TAU.items():
        d = J.diagrams(Q, charge=tau, phi=PHI, probe=gp, split=True)
        fx = exchange_flavour(tau, PHI)
        parts = {'s': d[1]['s'] + d[2]['s'], 'a': d[1]['a'] + d[2]['a'], 'm': d[3]['m']}
        diag = {'1': d[1]['s'] + d[1]['a'], '2': d[2]['s'] + d[2]['a'], '3': d[3]['m'],
                '4': 0, '5': 0, '6': 0}
        for XY, pcs in raw.items():
            ch = 's' if XY == 'ss' else ('a' if XY == 'aa' else 'm')
            for pc, val in pcs.items():
                parts[ch] = parts[ch] + fx[XY][pc] * val
                diag[pc[-1]] = diag[pc[-1]] + fx[XY][pc] * val
        res[fl] = {'total': (sum(parts.values()) / B).real.tolist(),
                   'dq': {k: (v / B).real.tolist() for k, v in parts.items()},
                   'diag': {k: (v / B).real.tolist() for k, v in diag.items()}}
    return res

out = {'M': M, 'grid': [nk, nl], 'names': GFF_NAMES, 'x': [], 'data': []}
t = time.time()
r0 = evaluate(0.0)
A0, c0 = r0['tot']['total']
print(f"Q = 0  ({time.time() - t:.0f}s):  A(0) = {A0:.5f}   cbar(0) = {c0:+.5f}")
print("       by diagram A: " + " ".join(f"{r0['tot']['diag'][k][0]:+.4f}" for k in '123456'))
print("       diquark content of A(0): " + "  ".join(f"{k} {100 * r0['tot']['dq'][k][0] / A0:5.1f}%" for k in 'sam')
      + "   (2611.19819: s 53.3%, a 34.0%, m 12.7%)")
print(f"       A^u(0) = {r0['u']['total'][0]:.4f}, A^d(0) = {r0['d']['total'][0]:.4f}, d/u = "
      f"{r0['d']['total'][0] / r0['u']['total'][0]:.3f}   (2611.19819: d/u = 0.529)")
out['Q0'] = r0

xs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
print("   x     |  A        J        D        S        cbar     B=2J-A")
for x in xs:
    t = time.time()
    r = evaluate(np.sqrt(x) * M)
    out['x'].append(x); out['data'].append(r)
    A, Jv, D, S, cb = r['tot']['total']
    print(f"  {x:5.3f}  | {A:+.4f}  {Jv:+.4f}  {D:+.4f}  {S:+.4f}  {cb:+.5f}  {2 * Jv - A:+.5f}   ({time.time() - t:.0f}s)")
    pathlib.Path('results').mkdir(exist_ok=True)
    json.dump(out, open(f'results/gff_{level}.json', 'w'), indent=1)

# forward limits from a quadratic fit over the four smallest x
def fwd(fl, key, idx, sub=None):
    y = [d[fl][key][sub][idx] if sub else d[fl][key][idx] for d in out['data'][:4]]
    return np.polyval(np.polyfit(xs[:4], y, 2), 0.0)
print("forward limits (quadratic fit, x <= 0.05)          | 2611.19819")
Jt, Dt, St = (fwd('tot', 'total', i) for i in (1, 2, 3))
print(f"   J(0) = {Jt:.4f}  (spin 1/2)                        | 0.500")
print(f"   D(0) = {Dt:.3f}                                     | -2.736")
print(f"   S(0) = {St:.3f}                                      | 0.401")
print("   diquark content of J(0): " + "  ".join(f"{k} {100 * fwd('tot', 'dq', 1, k) / Jt:5.1f}%" for k in 'sam')
      + "   | s 85.5%, m 14.3%, a 0.2%")
print("   diquark content of D(0): " + "  ".join(f"{k} {100 * fwd('tot', 'dq', 2, k) / Dt:5.1f}%" for k in 'sam')
      + "   | s 54.4%, a 40.4%, m 5.2%")
for i, nm in ((1, 'J'), (2, 'D')):
    u, d = fwd('u', 'total', i), fwd('d', 'total', i)
    print(f"   {nm}^u(0) = {u:+.4f}, {nm}^d(0) = {d:+.4f}, d/u = {d / u:+.3f}"
          + ("   | d/u = -0.024" if nm == 'J' else "   | d/u = 0.658"))
out['forward'] = dict(J=Jt, D=Dt, S=St)
json.dump(out, open(f'results/gff_{level}.json', 'w'), indent=1)
print(f"saved results/gff_{level}.json")
