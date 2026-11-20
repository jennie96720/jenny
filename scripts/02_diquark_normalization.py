"""Milestone 2: canonical normalisation of 0+ and 1+ diquark amplitudes, Eq. (A8).
   Target (arXiv:2611.19819 Eq. A12): g_{0+} = 14.94,  g_{1+} = 12.98.

Status / open convention question:
 * Taking Eqs. (A5),(A8) literally (Gamma^{0+} = g gamma_5 C t^0 H, full Tr_CDF over one
   colour component of H, Gammabar = C^dag Gamma^T C) gives g^2 < 0 and |g| about half the
   paper's value, in BOTH channels.
 * Sign: Segovia et al. (arXiv:1408.2919, Eq. B10) write C i gamma_5; the extra i flips g^2.
 * Factor 2 in g (4 in g^2): consistent with e.g. colour matrices eps_abc/sqrt2 together with a
   1/2 symmetry factor for the two identical quarks in the loop -- to be confirmed with the code
   used in the paper.  Only the product (g^2 x colour-flavour factors of the Faddeev kernel)
   is physical, so the kernel coefficients must use the same convention.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qdq.diquark import coupling

for ch, target in [('0+', 14.94), ('1+', 12.98)]:
    g, g2, cf = coupling(ch, nq=200, nz=24)
    print(f"{ch}: literal g^2 = {g2.real:9.3f}   2|g| = {2*abs(g):.3f}   "
          f"paper = {target}   rel. diff = {(2*abs(g)/target - 1)*100:+.2f}%")
