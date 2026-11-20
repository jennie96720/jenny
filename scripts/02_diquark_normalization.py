"""Milestone 2: canonical normalisation of 0+ and 1+ diquark amplitudes, Eq. (A8).
   Target (arXiv:2611.19819 Eq. A12): g_{0+} = 14.94,  g_{1+} = 12.98.

Convention (resolved in milestone 3):
 * Dirac structures C i gamma_5 and i gamma_mu C (Alkofer et al. nucl-th/0412046), unit-norm
   flavour matrices t_f (2611.19819), full colour trace Tr[H^T H] = 2 for one colour component.
   This gives g0+ = 7.54 and g1+ = 6.51 i (imaginary in our gamma representation).
 * The numbers quoted in 2611.19819 are 2x larger (14.94, 12.98): a different bookkeeping of
   colour/symmetry factors between normalisation and kernel.  What is physical is the product
   (kernel prefactor) x g_X g_Y: with OUR g and the kernel prefactor -4 of Alkofer et al., the
   Faddeev equation gives M_N = 1.174 GeV (paper 1.18) and scalar-only 1.455 GeV (ref. 1.46);
   using the paper's g/2 gives 1.179 and 1.459.  See scripts/03_faddeev_nucleon.py.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qdq.diquark import coupling

for ch, target in [('0+', 14.94), ('1+', 12.98)]:
    g, g2, cf = coupling(ch, nq=200, nz=24)
    print(f"{ch}: g^2 = {g2.real:9.3f}   g = {g:.4f}   2|g| = {2*abs(g):.3f}   "
          f"paper = {target}   rel. diff = {(2*abs(g)/target - 1)*100:+.2f}%")
