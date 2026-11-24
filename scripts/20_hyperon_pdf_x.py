"""x-dependence of the Lambda and Sigma^0 valence DFs from their Mellin moments (scripts/19), at the hadron
scale and at zeta_2 = 2 GeV (AO evolution), with the proton (same setup A) for reference.

Reconstruction as for the proton (qdq/pdf_x.py, scripts/11): positive-definite forms
N x^a (1-x)^b [exp(c x + d x^2)] fitted to the unit-normalised moments m = 1..M (M = the smallest
reliable m of the baryon's flavours, at most 10); the band is the envelope of the acceptable fits.
Error band of the inputs: the union of the envelopes for diquark mass spacings 0.10, 0.15, 0.20 GeV
per s quark (setups A10, A, A20); the line is the middle of setup A's envelope.
The x-shapes are illustrative (ill-posed inversion); x <~ 0.1 and x >~ 0.8 are extrapolations.

Usage: python scripts/20_hyperon_pdf_x.py   -> results/hyperon_pdf_x.json, results/hyperon_pdf_valence.png
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qdq.pdf_x import band

H = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results/hyperon_pdf_highm.json'))
fac = np.array(H['valence_factor_2GeV'])
x = np.linspace(0.002, 0.998, 499)
MMAX = 10


def moments(key, q):
    o = H[key][q]
    return np.array(o['moments']), o['m_reliable']


def curves(b, setups=('A', 'A10', 'A20')):
    """{scale: {q: (line, lo, hi)}} and per-setup info."""
    out, info = {}, {}
    avail = [s for s in setups if f'{s}/{b}' in H]
    for scale in ('zeta_H', 'zeta_2'):
        out[scale] = {}
        for q in (('u', 's') if b != 'p' else ('u', 'd')):
            env = []
            for s in avail:
                mom, _ = moments(f'{s}/{b}', q)
                M = min(MMAX, min(moments(f'{s}/{b}', qq)[1] for qq in (('u', 's') if b != 'p' else ('u', 'd'))))
                mm = mom[:M + 1] * (fac[:M + 1] if scale == 'zeta_2' else 1)
                lo, hi, fits = band(mm, x, chi2_max=3.0)
                env.append((s, lo, hi))
                info[f'{s}/{b}/{q}/{scale}'] = dict(M=M, fits={f.form: round(float(f.chi2), 3) for f in fits})
            lo = np.min([e[1] for e in env], 0)
            hi = np.max([e[2] for e in env], 0)
            a = [e for e in env if e[0] == 'A'][0]
            out[scale][q] = (0.5 * (a[1] + a[2]), lo, hi)
    return out, info


def main():
    res, info = {}, {}
    for b in ('p', 'Lambda', 'Sigma0'):
        res[b], inf = curves(b)
        info.update(inf)
    json.dump({'x': x.tolist(), 'info': info,
               'xq': {b: {sc: {q: [c.tolist() for c in v] for q, v in d.items()} for sc, d in r.items()}
                      for b, r in res.items()}},
              open('results/hyperon_pdf_x.json', 'w'))
    for k, v in info.items():
        if k.startswith('A/'):
            print(k, v)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    col = {'u': '#2a78d6', 's': '#eb6834'}
    lab = {'u': 'l = u = d', 's': 's'}
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.2), dpi=150)
    for row, b, name in ((0, 'Lambda', r'$\Lambda$'), (1, 'Sigma0', r'$\Sigma^0$')):
        for c, sc, title in ((0, 'zeta_H', r'hadron scale $\zeta_H$'), (1, 'zeta_2', r'$\zeta_2 = 2$ GeV')):
            ax = axs[row, c]
            ax.set_facecolor('#fcfcfb')
            for q in ('u', 's'):
                line, lo, hi = res[b][sc][q]
                ax.fill_between(x, lo, hi, color=col[q], alpha=0.30, lw=0)
                ax.plot(x, line, color=col[q], lw=2, label=f'{lab[q]} in {name}')
            pu, pd = res['p'][sc]['u'][0], res['p'][sc]['d'][0]
            ax.plot(x, 0.5 * pu, color='#52514e', lw=1, ls='--', label=r'proton $u/2$')
            ax.plot(x, pd, color='#52514e', lw=1, ls=':', label=r'proton $d$')
            ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
            ax.set_xlabel('$x$'); ax.set_ylabel('$x\\,q(x)$')
            ax.set_title(f'{name}, {title}', color='#0b0b0b', fontsize=10)
            ax.grid(color='#e6e5e1', lw=0.6); ax.set_axisbelow(True)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            if c == 0:
                ax.legend(frameon=False, fontsize=8)
        # ratio s/l at the hadron scale, central (setup A) and the spread of setups
        ax = axs[row, 2]
        ax.set_facecolor('#fcfcfb')
        rs = []
        for s in ('A10', 'A', 'A20'):
            if f'{s}/{b}' not in H:
                continue
            r = {}
            for q in ('u', 's'):
                mom, _ = moments(f'{s}/{b}', q)
                M = info[f'{s}/{b}/{q}/zeta_H']['M']
                lo, hi, _ = band(mom[:M + 1], x, chi2_max=3.0)
                r[q] = 0.5 * (lo + hi)
            rs.append((s, r['s'] / r['u']))
        sel = (x > 0.1) & (x < 0.8)                       # outside: extrapolation of the fit forms
        for s, rr in rs:
            ax.plot(x[sel], rr[sel], color='#0b0b0b' if s == 'A' else '#52514e', lw=2 if s == 'A' else 1,
                    ls='-' if s == 'A' else ('--' if s == 'A10' else ':'),
                    label={'A': 'spacing 0.15 GeV (A)', 'A10': '0.10 GeV', 'A20': '0.20 GeV'}[s])
        ax.axhline(1, color='#e6e5e1', lw=1)
        ax.set_xlim(0, 0.85); ax.set_ylim(0, 3 if b == 'Lambda' else 2)
        ax.set_xlabel('$x$'); ax.set_ylabel('$s(x)/l(x)$')
        ax.set_title(f'{name}: $s/l$ at $\\zeta_H$ (middle of fit envelope)', color='#0b0b0b', fontsize=10)
        ax.grid(color='#e6e5e1', lw=0.6); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Valence DFs of $\\Lambda$ and $\\Sigma^0$, quark + diquark Faddeev amplitude; bands: moment '
                 'reconstruction $\\cup$ diquark mass spacing 0.10-0.20 GeV per s quark', fontsize=10, color='#52514e')
    fig.tight_layout()
    fig.savefig('results/hyperon_pdf_valence.png')
    print("saved results/hyperon_pdf_x.json, results/hyperon_pdf_valence.png")


if __name__ == '__main__':
    main()
