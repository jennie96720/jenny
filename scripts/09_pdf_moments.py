"""PDF step 1: Mellin moments <x^m> of the proton's hadron-scale valence u and d distributions from
the momentum-dependent quark+diquark Faddeev amplitude (qdq/pdf.py).

Checks printed:
  m = 0  ->  u, d numbers 2, 1 (exact by the Ward identity; normalisation from dlambda/dM only)
  m = 1  ->  momentum fractions = gravitational A^u(0), A^d(0) (0.6893, 0.3144 on this grid)
  imaginary parts ~ 0;  stability under the UV cutoff kmax (10 vs 15 GeV).
Diagram 2 (probe on the diquark) is printed as the diquark light-front distribution <y^m>_D; the
proton moments use the quark-in-diquark DF moments <z^m> (step 2).  Until then the placeholder
<z^m> = 2^-m is used, so only m = 0, 1 of the totals are final.
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qdq.faddeev import NucleonFaddeev
from qdq.current import NucleonCurrent
from qdq.exchange import ExchangeCurrent
from qdq.pdf import NucleonPDFMoments

np.set_printoptions(precision=5, suppress=True, linewidth=150)
F = NucleonFaddeev(g=(14.94 / 2, 1j * 12.98 / 2), nr=16, nz=8, ny=8, ncheb=6)
F.solve_mass(1.17, 1.19, verbose=False)
J = NucleonCurrent(F, nk=48, nz=32, ny=24)
X = ExchangeCurrent(J, nk=(12, 6, 6), nl=(12, 6, 6, 6))
P = np.array([0, 0, 0, 1j * F.M])
out = {'M': F.M, 'm': list(range(7))}
for kmax in (10.0, 15.0):
    pm = NucleonPDFMoments(J, X, mmax=6, kmax=kmax)
    r = pm.moments(P)
    print(f"kmax = {kmax} GeV")
    for fl in 'ud':
        print(f"  <x^m>_{fl} (m=0..6):", r[fl]['total'].real, f"  max|Im| {np.abs(r[fl]['total'].imag).max():.0e}")
        for k in '12456':
            print(f"      diagram {k}:", np.real(r[fl][k]))
    print("  diquark light-front distribution <y^m>_D (per flavour count):")
    for fl in 'ud':
        print(f"      {fl}: scalar", r[fl]['diquark_y']['s'].real, " axial", r[fl]['diquark_y']['a'].real)
    out[f'kmax{int(kmax)}'] = {fl: {k: (np.real(v).tolist() if not isinstance(v, dict) else
                                       {c: np.real(w).tolist() for c, w in v.items()})
                                    for k, v in r[fl].items()} for fl in 'ud'}
pathlib.Path('results').mkdir(exist_ok=True)
json.dump(out, open('results/pdf_moments_step1.json', 'w'), indent=1)
print("saved results/pdf_moments_step1.json")
