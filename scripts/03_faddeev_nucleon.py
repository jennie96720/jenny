"""Milestone 3: nucleon quark+diquark Faddeev equation -> core mass.

Checks (same parameters, m_{0+}=0.79, m_{1+}=0.89 GeV, w^2 = m^2/2):
  * full nucleon:          M_N = 1.18 GeV   (arXiv:2611.19819; Alkofer et al. Set B)
  * scalar diquark only:   M_N = 1.46 GeV   (Alkofer et al. nucl-th/0412046, Sect. 2.3)

Couplings are computed self-consistently from the canonical normalisation
(scripts/02) with the SAME Dirac structures used in the kernel; with our gamma-matrix
representation g_{1+} comes out purely imaginary, and that phase must be kept.
Usage:  python3 scripts/03_faddeev_nucleon.py [nr nz ny ncheb]
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qdq.faddeev import NucleonFaddeev, flavour_coeffs
from qdq.diquark import coupling

nr, nz, ny, nc = (int(a) for a in sys.argv[1:5]) if len(sys.argv) >= 5 else (16, 8, 8, 4)
grid = dict(nr=nr, nz=nz, ny=ny, ncheb=nc)
gs, ga = coupling('0+')[0], coupling('1+')[0]
print(f"couplings: g0+ = {gs:.4f},  g1+ = {ga:.4f}")
print("flavour coefficients:", {k: round(v, 4) for k, v in flavour_coeffs().items()})
print(f"grid: {grid}")
print("scalar diquark only (target 1.46 GeV):")
Ms = NucleonFaddeev(g=(gs, ga), channels='s', **grid).solve_mass(1.40, 1.50)
print("scalar + axial-vector (target 1.18 GeV):")
Mf = NucleonFaddeev(g=(gs, ga), channels='sa', **grid).solve_mass(1.15, 1.20)
print(f"RESULT: scalar-only M = {Ms:.4f} GeV,  full M_N = {Mf:.4f} GeV")
