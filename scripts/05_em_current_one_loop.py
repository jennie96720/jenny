"""Milestone 5b: nucleon EM current, one-loop diagrams 1-3 (arXiv:2507.13484 construction).

Checks performed:
  1. every vertex Ansatz satisfies its Ward-Green-Takahashi identity (machine precision);
  2. quark loop inside the diquark reproduces the elastic scalar-diquark charge 2K_mu (fixes the
     sign convention of loop-derived vertices) and gives the 0+<->1+ transition phase/sign;
  3. Ward identity for the baryon-number current at Q=0:
       diagrams 1+2 (baryon charges)  ==  -Psibar dG0/dP Psi   (independent finite difference)
     and the full charge from the Faddeev eigenvalue,  J^B = -dlambda/dP  Psibar G0 Psi,
     which normalises the amplitude canonically WITHOUT the two-loop diagrams.
Output: normalised contributions of diagrams 1-3 to F1, F2 (and G_E, G_M) of p and n.
The two-loop diagrams 4-6 (milestone 5c) must supply  F1p(0) -> 1  and  F1n(0) -> 0.
"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qdq.faddeev import NucleonFaddeev, dot
from qdq.current import (NucleonCurrent, Vertices, EQ, EB, diquark_triangle, transition_kappas)
from qdq.quark import S
from qdq.diquark import coupling

print("1) Ward-Green-Takahashi identities of the vertices (random complex momenta)")
rng = np.random.default_rng(1)
V = Vertices(M=1.18)
l2 = rng.normal(size=(6, 4)) * 0.5 + 1j * rng.normal(size=(6, 4)) * 0.3
Q = np.array([0.1, -0.3, 0.4, 0.2]) + 0j
l1 = l2 + Q
Q2 = dot(Q, Q)
chk = {
    'quark  i Q.chi = S(l2)-S(l1)':
        (1j * np.einsum('m,nmab->nab', Q, V.chi_quark(l1, l2)), S(l2) - S(l1)),
    'scalar Q.chi0 = F[D(l2)-D(l1)]':
        (np.einsum('m,nm->n', Q, V.chi_s(l1, l2)),
         V.Fj(Q2, V.p['r_sc']) * (V.D0(dot(l2, l2)) - V.D0(dot(l1, l1)))),
    'axial  Q.chi1 = F[D(l2)-D(l1)]':
        (np.einsum('m,nmab->nab', Q, V.chi_a(l1, l2)), V.Fj(Q2, V.p['r_av']) * (V.D1mat(l2) - V.D1mat(l1))),
    'transition Q.chi01 = 0': (np.einsum('m,nma->na', Q, V.chi_01(l1, l2)), 0),
}
for name, (a, b) in chk.items():
    print(f"   {name:34s} max|diff| = {np.abs(a - b).max():.1e}")

print("2) quark loop inside the diquark (own canonically normalised couplings)")
g = (coupling('0+')[0], coupling('1+')[0])
K = np.array([0, 0, 0, 1j * 0.79])
Ta, Tb = diquark_triangle('s', 's', K, K, g)
print(f"   elastic 0+, Q=0, on shell: -(T_a+T_b)/2 = {(-(Ta + Tb) / 2)[3]:.5f}   expected 2K_4 = {2 * K[3]:.5f}")
r = transition_kappas(g, M=1.18, Qs=0.05)
print(f"   transition: kappa_01 = {r['sa']['kappa']:.3f}, kappa_10 = {r['as']['kappa']:.3f}"
      f"  (used: |kappa| = 2.64 of 2507.13484, phase -i, chi^10 = +chi^01)")

print("3) Faddeev amplitude (N_cheb = 6) and baryon-number Ward identity")
F = NucleonFaddeev(g=(14.94 / 2, 1j * 12.98 / 2), nr=16, nz=8, ny=8, ncheb=6)
F.solve_mass(1.17, 1.19, verbose=False)
M = F.M
J = NucleonCurrent(F, nk=32, nz=16, ny=12)
B, F1_G0, dlam, n0 = J.normalisation()
d = J.diagrams(0.0, charge=EB, which=(1, 2))
print(f"   M_N = {M:.4f} GeV,  dlambda/dM = {dlam.real:.4f} GeV^-1")
print(f"   diagrams 1+2 (baryon charges)  = {(d[1][0] + d[2][0]).imag:.7f} i")
print(f"   -Psibar dG0/dP Psi (fin. diff.) = {F1_G0.imag:.7f} i")
print(f"   full baryon charge (eigenvalue) = {B.imag:.7f} i   -> impulse share {((d[1][0] + d[2][0]) / B).real:.4f}")

print("4) normalised contributions of diagrams 1-3")
print("    x=Q^2/M^2  N   |  F1: d1      d2      d3      sum  |  F2: d1      d2      d3      sum  |  G_E    G_M  (1-3 only)")
for x in [0.0, 0.02, 0.5, 1.0, 2.0, 3.0]:
    Qm = np.sqrt(x) * M
    for nm, phi in [('p', np.array([1., 0])), ('n', np.array([0., 1]))]:
        dd = J.diagrams(Qm, charge=EQ, phi=phi)
        v = {k: (val / B).real for k, val in dd.items()}
        F1 = [v[k][0] for k in (1, 2, 3)]
        F2 = [v[k][1] if len(v[k]) > 1 else np.nan for k in (1, 2, 3)]
        s1, s2 = sum(F1), sum(F2)
        GE, GM = s1 - x / 4 * s2, s1 + s2
        print(f"    {x:5.2f}     {nm}   | " + " ".join(f"{a:+.4f}" for a in F1) + f" {s1:+.4f} | "
              + " ".join(f"{a:+.4f}" for a in F2) + f" {s2:+.4f} | {GE:+.4f} {GM:+.4f}")
print("   targets for diagrams 4-6:  F1p(0) = 1 - (sum above),  F1n(0) = -(sum above)")
