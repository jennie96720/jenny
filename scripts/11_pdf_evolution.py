"""PDF steps 3-4: AO evolution of the hadron-scale proton DF moments and x-dependent valence DFs.

Input: results/pdf_moments_highm.json (scripts/12: converged grids, m <= 9-10) if present, else
results/pdf_moments_step2.json (scripts/10, coarse two-loop grid, m <= 6).  Output: results/pdf_evolved.json,
results/pdf_valence.png, and comparison tables against
  - arXiv:2505.10663 Table 4 (SCI, zeta_2 = 2 GeV: valence, glue, sea),
  - EPJC 84 (2024) 739 Table 2 (zeta_3 = 3.097 GeV: SCI and the realistic CSM DFs of its Ref. [21]).
Valence evolution depends only on <x>_val(zeta)/<x>_val(zeta_H) (0.477 at 2 GeV, as in the papers);
glue and total sea are robust; the sea flavour split depends on the threshold treatment (see
qdq/evolution.py) and is indicative only."""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qdq.evolution import AOEvolution, ZETA_H
from qdq.pdf_x import band

Z2, Z3 = 2.0, 3.097
hm = pathlib.Path('results/pdf_moments_highm.json')
if hm.exists():
    h = json.load(open(hm))
    H = {fl: np.array(h[fl]['moments'][:h[fl]['m_reliable'] + 1]) for fl in 'ud'}
    print("hadron-scale moments from", hm, {fl: len(H[fl]) - 1 for fl in 'ud'})
else:
    s2 = json.load(open('results/pdf_moments_step2.json'))
    H = {fl: np.array(s2[fl]['3par']) for fl in 'ud'}      # <x^m>, m = 0..6, at zeta_H
MH = max(len(H['u']), len(H['d'])) - 1
ev = AOEvolution()
out = {'zeta_H': ZETA_H, 'hadron_scale': {fl: H[fl].tolist() for fl in 'ud'}}

fac = {z: np.array([1.0] + [ev.valence_factor(n, z) for n in range(1, MH + 1)]) for z in (Z2, Z3)}
print(f"zeta_H = {ZETA_H} GeV;  <x>_val factor: {fac[Z2][1]:.4f} (2 GeV), {fac[Z3][1]:.4f} (3.097 GeV)")
for z in (Z2, Z3):
    out[f'valence_{z}'] = {fl: (H[fl] * fac[z][:len(H[fl])]).tolist() for fl in 'ud'}

print("\nValence moments (%), zeta_3 = 3.097 GeV  [EPJC 84:739 Table 2]")
T2 = {'SCI': {'u': [30.2, 8.4, 3.1], 'd': [13.9, 3.7, 1.3]},
      'Realistic [21]': {'u': ['30.4(1.3)', '7.7(5)', '2.5(2)'], 'd': ['13.8(6)', '3.2(2)', '0.9(1)']}}
print(f"{'':10s} {'this work (q(qq), realistic)':>30s} {'SCI':>18s} {'Realistic [21]':>28s}")
for fl in 'ud':
    for n in (1, 2, 3):
        v = 100 * H[fl][n] * fac[Z3][n]
        print(f"  <x^{n}>_{fl}   {v:22.2f}        {T2['SCI'][fl][n-1]:12}   {T2['Realistic [21]'][fl][n-1]:>18}")
iv = [100 * (H['u'][n] - H['d'][n]) * fac[Z3][n] for n in (1, 2, 3)]
print(f"  isovector u-d: {np.round(iv, 2)}  (SCI 16.4, 4.7, 1.8; Ref. [21]: 16.6(7), 4.5(3), 1.6(1))")

print("\nzeta_2 = 2 GeV, all parton species (%)  [2505.10663 Table 4, SCI proton]")
T4 = {'u': [32.7, 9.47, 3.58], 'd': [15.0, 4.15, 1.54], 'g': [41.4, 2.67, 0.500],
      'S_u': [3.29, 0.175, 0.0267], 'S_d': [4.19, 0.258, 0.0425], 'S_s': [2.48, 0.136, 0.0213],
      'S_c': [0.850, 0.0404, 0.0060]}
keys = ['u', 'd', 'g', 'S_u', 'S_d', 'S_s', 'S_c']
print('         ' + ''.join(f"{k:>16s}" for k in keys))
out['zeta2_all'] = {}
for n in (1, 2, 3):
    o = ev.evolve(n, {fl: H[fl][n] for fl in 'ud'}, Z2)
    out['zeta2_all'][n] = {k: o[k] for k in keys}
    print(f"  <x^{n}>  " + ''.join(f"{100*o[k]:8.3g} ({T4[k][n-1]:<5})" for k in keys))
    if n == 1:
        tot = sum(o[k] for k in keys)
        print(f"          momentum sum {100*tot:.2f} %;  total sea {100*sum(o[k] for k in keys[3:]):.2f} % (SCI 10.81)")

# ---- x-dependence: bands of positive-definite reconstructions ------------------------------
x = np.linspace(0.002, 0.998, 499)
curves = {}
for tag, mom in (('zeta_H', H), ('zeta_2', {fl: H[fl] * fac[Z2][:len(H[fl])] for fl in 'ud'})):
    curves[tag] = {}
    for fl in 'ud':
        lo, hi, fits = band(mom[fl], x, chi2_max=3.0)
        curves[tag][fl] = (lo, hi)
        print(f"{tag} {fl}: fits " + ', '.join(f"{q.form} chi2/dof {q.chi2:.2f}" for q in fits))
out['x'] = x.tolist()
out['xq_band'] = {t: {fl: [c[0].tolist(), c[1].tolist()] for fl, c in v.items()} for t, v in curves.items()}
json.dump(out, open('results/pdf_evolved.json', 'w'), indent=1)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e'})
col = {'u': '#2a78d6', 'd': '#eb6834'}
fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=False)
for ax, (tag, title) in zip(axs, (('zeta_H', r'hadron scale $\zeta_H$'), ('zeta_2', r'$\zeta_2 = 2$ GeV (AO evolution)'))):
    ax.set_facecolor('#fcfcfb')
    for fl, lab in (('u', r'$x\,u_V(x)$'), ('d', r'$x\,d_V(x)$')):
        lo, hi = curves[tag][fl]
        ax.fill_between(x, lo, hi, color=col[fl], alpha=0.35, lw=0, label=lab)
        ax.plot(x, 0.5 * (lo + hi), color=col[fl], lw=2)
        i = np.argmax(hi)
        ax.annotate(fl + r'$_V$', (x[i], hi[i]), xytext=(6, 4), textcoords='offset points', color='#0b0b0b')
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    ax.set_xlabel('$x$'); ax.set_title(title, color='#0b0b0b', fontsize=11)
    ax.grid(color='#e6e5e1', lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
axs[0].set_ylabel('$x\\,q(x)$')
axs[0].legend(frameon=False)
fig.suptitle(f'Proton valence DFs, quark + diquark Faddeev amplitude (band: reconstruction from $m\\leq{MH}$ moments)',
             fontsize=10, color='#52514e')
fig.tight_layout()
fig.savefig('results/pdf_valence.png')
print("saved results/pdf_evolved.json, results/pdf_valence.png")
