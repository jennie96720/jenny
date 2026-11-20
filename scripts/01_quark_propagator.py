"""Milestone 1: algebraic quark propagator. Checks against arXiv:2611.19819 App. A.1:
   m_q = 5.08 MeV,  Euclidean constituent mass M^E (p^2 = M^2(p^2)) = 0.33 GeV."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import brentq
from qdq.dirac import check
from qdq.quark import mass_function, LAMBDA, PARAMS

check()
print(f"current mass m_q = mbar*lambda = {PARAMS['mbar']*LAMBDA*1e3:.2f} MeV   (paper: 5.08 MeV)")
ME = brentq(lambda p: p**2 - mass_function(p**2).real**2, 0.05, 2.0)
print(f"Euclidean constituent mass M^E = {ME:.3f} GeV          (paper: 0.33 GeV)")
print(f"M(0) = {mass_function(0.0).real:.3f} GeV")
for p2 in [0.1, 1.0, 10.0, 100.0]:
    print(f"  M(p^2={p2:6.1f} GeV^2) = {mass_function(p2).real:.4f} GeV")
