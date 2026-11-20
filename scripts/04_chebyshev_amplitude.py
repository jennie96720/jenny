"""Milestone 5a: Chebyshev representation of the nucleon Faddeev amplitude.

Checks:
  1. ncheb = nz reproduces the Nystrom eigenvalue exactly (similarity transform);
  2. M_N converges rapidly in the number of Chebyshev moments;
  3. the truncated-Chebyshev amplitude agrees with the full Nystrom amplitude;
  4. relative size of the moments f_{c,n}  (which structures / moments matter).
Couplings: paper values / 2 (see scripts/02), i.e. g0+ = 7.47, g1+ = 6.49 i.
"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qdq.faddeev import NucleonFaddeev, NAMES

g = (14.94 / 2, 1j * 12.98 / 2)
grid = dict(nr=16, nz=8, ny=8)

print("1) eigenvalue at M = 1.18 GeV")
for nc in [None, grid['nz']]:
    lam = NucleonFaddeev(g=g, ncheb=nc, **grid).eigen(1.18)[0]
    print(f"   ncheb = {str(nc):4s}: lambda = {lam.real:.12f}")

print("2) M_N vs number of Chebyshev moments")
sols = {}
for nc in [1, 2, 3, 4, 6, None]:
    F = NucleonFaddeev(g=g, ncheb=nc, **grid)
    t = time.time()
    M = F.solve_mass(1.17, 1.19, verbose=False)
    sols[nc] = F
    print(f"   ncheb = {str(nc):4s}: M_N = {M:.5f} GeV   ({time.time() - t:5.1f} s)")

print("3) amplitude: ncheb = 4 vs Nystrom, on the Nystrom nodes (relative to max|s1|)")
ref, F4 = sols[None], sols[4]
R, Z = np.meshgrid(ref.r, ref.z, indexing='ij')
lv = np.stack([0 * R, 0 * R, R * np.sqrt(1 - Z**2), R * Z], -1)
P = np.array([0, 0, 0, 1j * ref.M])
fa = ref.scalar_functions(lv, P)
fb = F4.scalar_functions(lv, np.array([0, 0, 0, 1j * F4.M]))
scale = np.abs(fa[..., 0]).max()
for c, nm in enumerate(NAMES):
    print(f"   {nm}: max|f| = {np.abs(fa[..., c]).max() / scale:7.4f}   "
          f"max|diff| = {np.abs(fa[..., c] - fb[..., c]).max() / scale:.1e}")

print("4) size of Chebyshev moments (Nystrom solution, max over r, relative to s1_0)")
f = ref.f                                           # (c, n, r)
print("        " + "".join(f"  n={n:<5d}" for n in range(6)))
for c, nm in enumerate(NAMES):
    print(f"   {nm}: " + "".join(f"{np.abs(f[c, n]).max() / np.abs(f[0, 0]).max():9.4f}" for n in range(6)))
