"""PDF step 2: quark-in-diquark DFs q^D(z) = n [phi_D(z)]^2 (EPJC 84:739 App. B; 2505.10663 App. A)
and the completed proton moments <x^m>_{u,d} = step-1 diagrams 1, 4-6 + sum_D <y^m>_D <z^m>_{q/D}.

phi_D: leading-twist DA of the 0+ / 1+ diquark from our amplitudes (qdq/diquark_df.py), via its
Euclidean moments (kmax-stable through <xi^6>) and a 3-parameter interpolation.  Needs
results/pdf_moments_step1.json (scripts/09)."""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qdq.diquark_df import quark_in_diquark, da_moments, _xi_moments_from_x

np.set_printoptions(precision=5, suppress=True, linewidth=150)
MM = 6
print("DA xi-moments vs UV cutoff (stability):")
for ch in ('0+', '1+'):
    for kmax in (15., 20., 30.):
        xm, _ = da_moments(ch, mmax=8, kmax=kmax)
        print(f"  {ch} kmax {kmax:4.0f}: <xi^2,4,6,8> =", _xi_moments_from_x(xm).real[[2, 4, 6, 8]])
qd = quark_in_diquark(mmax=MM)
for ch, v in qd.items():
    d = v['da']
    print(f"{ch}: alpha = {d.alpha:.4f}, a2, a4 = {d.a},  fit <xi^2,4,6,8> = {v['xi_fit'][[2, 4, 6, 8]]}")
    print(f"     <z^m>_q/D (3-par) = {v['z']}\n     <z^m>_q/D (2-par) = {v['z_2par']}")
s1 = json.load(open('results/pdf_moments_step1.json'))['kmax15']
out = {'z': {ch: qd[ch]['z'].tolist() for ch in qd}, 'z_2par': {ch: qd[ch]['z_2par'].tolist() for ch in qd},
       'da': {ch: dict(alpha=qd[ch]['da'].alpha, a=qd[ch]['da'].a.tolist()) for ch in qd}}
for fl in 'ud':
    r = s1[fl]
    ys, ya = np.array(r['diquark_y']['s'])[:MM + 1], np.array(r['diquark_y']['a'])[:MM + 1]
    rest = sum(np.array(r[k])[:MM + 1] for k in '1456')
    tot = {}
    for tag, zs, za in (('3par', qd['0+']['z'], qd['1+']['z']), ('2par', qd['0+']['z_2par'], qd['1+']['z_2par'])):
        tot[tag] = rest + ys * zs + ya * za
    print(f"<x^m>_{fl} (m = 0..{MM}):", tot['3par'], "\n      2-par DA fit:", tot['2par'],
          "\n      diagram 2 now:", ys * qd['0+']['z'] + ya * qd['1+']['z'])
    out[fl] = {k: v.tolist() for k, v in tot.items()}
json.dump(out, open('results/pdf_moments_step2.json', 'w'), indent=1)
print("saved results/pdf_moments_step2.json")
