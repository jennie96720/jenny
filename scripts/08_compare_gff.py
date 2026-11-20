"""Compare our gravitational form factors (results/gff_<level>.json) with a reference data set,
e.g. P. Cheng's results for arXiv:2611.19819 exported from Mathematica as JSON:
  {"central": {"AGPengtot": [[Q2, value], ...], "AGPengs": ..., ...}, "m0.95": {...}, "m1.05": {...}}
(The reference data are not stored in this repository.)

Usage: python3 scripts/08_compare_gff.py <reference.json> [coarse|fine]
"""
import sys, json, pathlib
import numpy as np

ref = json.load(open(sys.argv[1]))
level = sys.argv[2] if len(sys.argv) > 2 else 'coarse'
ours = json.load(open(pathlib.Path(__file__).resolve().parents[1] / f'results/gff_{level}.json'))
M = ours['M']
IDX = {'A': 0, 'J': 1, 'D': 2, 'cbar': 4}
REFKEY = {'A': 'AGPeng', 'J': 'JGPeng', 'D': 'DGPeng', 'B': 'BGPeng', 'cbar': 'CGPeng'}

def ref_at(name, part, Q2, which='central'):
    tab = ref[which].get(REFKEY[name] + part)
    if tab is None:
        return np.nan
    q, v = np.array(tab).T
    return np.interp(Q2, q, v, left=np.nan, right=np.nan)

def our(d, name, fl, part=None):
    blk = d[fl]
    vals = blk['dq'][part] if part else blk['total']
    if name == 'B':
        return 2 * vals[1] - vals[0]
    return vals[IDX[name]]

rows = [(0.0, ours['Q0'])] + list(zip(ours['x'], ours['data']))
for name in ['A', 'J', 'B', 'D', 'cbar']:
    print(f"\n{name}:  Q^2[GeV^2] | total ours / ref | s ours / ref | a ours / ref | m ours / ref | u ours / ref | d ours / ref")
    for x, d in rows:
        if x == 0 and name not in ('A',):
            continue
        Q2 = x * M**2
        Qr = max(Q2, 1e-4)
        cells = []
        for fl, part, rk in [('tot', None, 'tot'), ('tot', 's', 's'), ('tot', 'a', 'a'), ('tot', 'm', 'm'),
                             ('u', None, 'u'), ('d', None, 'd')]:
            if x == 0 and name == 'A':
                o = d[fl]['dq'][part][0] if part else d[fl]['total'][0]
            else:
                o = our(d, name, fl, part)
            cells.append(f"{o:+.4f}/{ref_at(name, rk, Qr):+.4f}")
        print(f"   {Q2:7.4f}   | " + " | ".join(cells))
