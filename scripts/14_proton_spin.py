"""Gluon helicity and proton spin decomposition from our hadron-scale results (qdq/spin.py).

Inputs: Delta u, Delta d (results/pdf_helicity_moments.json, scripts/13); <x>_u, <x>_d
(results/pdf_moments_highm.json, scripts/12); J^u, J^d (results/gff_coarse.json, scripts/07, extrapolated
to Q^2 = 0).  Light-front budget 1/2 = a_I/2 + l_q + DeltaG + l_g as in EPJC 84:739 Sect. 5 and
Cheng et al. PLB 844 (2023) 138074 (zeta_C^2 = 3 GeV^2); Ji budget J_q = Delta q/2 + L_q from the GFFs.
Validation (scratch, recorded in README): with their inputs we reproduce Cheng et al. 2023
(a_0 = 0.65: L_q = -0.027, DeltaG = 1.29-1.34 cf. 1.23, a_0^E = 0.33-0.34 cf. 0.35(2)).
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qdq.spin import decomposition, wandzura_wilczek_lf
from qdq.evolution import AOEvolution, ZETA_H_NOMINAL

hel = json.load(open('results/pdf_helicity_moments.json'))
unp = json.load(open('results/pdf_moments_highm.json'))
gff = json.load(open('results/gff_coarse.json'))
Du, Dd = hel['u']['moments'][0], hel['d']['moments'][0]
xu, xd = unp['u']['moments'][1], unp['d']['moments'][1]
gA, aI = Du - Dd, Du + Dd
# J^q(Q^2 -> 0): linear extrapolation from the two smallest Q^2 points
x0, x1 = gff['x'][0], gff['x'][1]
J = {fl: gff['data'][0][fl]['total'][1] + (gff['data'][0][fl]['total'][1] - gff['data'][1][fl]['total'][1]) * x0 / (x1 - x0)
     for fl in ('u', 'd')}
out = {'inputs': dict(Delta_u=Du, Delta_d=Dd, g_A=gA, a_I=aI, x_u=xu, x_d=xd, J_u=J['u'], J_d=J['d'])}
print(f"zeta_H: g_A = {gA:.4f}, a_I = Delta u + Delta d = {aI:.4f} ({aI / gA:.3f} g_A)   [Cheng 2023: a_0 = 0.65(2) = 0.52 g_A]")

lf = {fl: wandzura_wilczek_lf(x, D) for fl, x, D in (('u', xu, Du), ('d', xd, Dd))}
ji = {fl: J[fl] - 0.5 * D for fl, D in (('u', Du), ('d', Dd))}
B = {fl: 2 * J[fl] - x for fl, x in (('u', xu), ('d', xd))}
print("hadron-scale quark OAM by flavour:")
print(f"   light-front (Wandzura-Wilczek, EPJC Eq. 46): l_u = {lf['u']:.3f}, l_d = {lf['d']:.3f}, sum {lf['u'] + lf['d']:.3f} (= 1/2 - a_I/2)")
print(f"   Ji (our GFFs J^q):                         L_u = {ji['u']:.3f}, L_d = {ji['d']:.3f}, sum {ji['u'] + ji['d']:.3f} (J = {J['u'] + J['d']:.3f})")
print(f"   difference = B_q(0)/2:  B_u = {B['u']:.3f}, B_d = {B['d']:.3f}  (sum {B['u'] + B['d']:.3f}; 0 by the spin sum rule)")
out['zeta_H'] = dict(lf=lf, ji=ji, B=B)

rows = []
for lab, z in (('zeta_C (sqrt 3 GeV)', np.sqrt(3.0)), ('2 GeV', 2.0), ('sqrt 10 GeV', np.sqrt(10.0))):
    d0 = decomposition(aI, z)
    d1 = decomposition(aI, z, AOEvolution(zeta_H=ZETA_H_NOMINAL))
    rows.append((lab, d0, d1))
    out[lab] = {k: (d0[k], d1[k]) for k in ('DeltaG', 'l_q', 'l_g', 'aIE', 'J_q', 'J_g', 'alpha')}
print("\nlight-front spin budget  1/2 = a_I/2 + l_q + DeltaG + l_g  (zeta_H = 0.3266 | nominal 0.331):")
print(f"{'':20s} {'a_I/2':>8s} {'l_q':>16s} {'DeltaG':>16s} {'l_g':>16s} {'a_I^E':>16s}")
for lab, d0, d1 in rows:
    f = lambda k: f"{d0[k]:7.3f}|{d1[k]:<7.3f}"
    print(f"{lab:20s} {aI / 2:8.3f} {f('l_q'):>16s} {f('DeltaG'):>16s} {f('l_g'):>16s} {f('aIE'):>16s}")
print("References: COMPASS a_0(zeta^2 = 3 GeV^2) = 0.32(7); Cheng 2023 a_0^E = 0.35(2), DeltaG = 1.23, L_q = -0.027, "
      "L_g = -1.03 (zeta_C); lattice (chiQCD) glue spin S_G = 0.251(47)(16) at mu^2 = 10 GeV^2 (MS-bar, Coulomb gauge).")
json.dump(out, open('results/proton_spin.json', 'w'), indent=1, default=float)
print("saved results/proton_spin.json")
