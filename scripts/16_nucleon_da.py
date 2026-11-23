"""Proton leading-twist DA from the quark+diquark Faddeev amplitude: f_N and first moments
(qdq/nucleon_da.py), as a function of the UV cutoff Lambda on the diquark-loop momentum.

Structural checks printed for every run: residual of R ~ n.gamma H_+ Lambda_+ (must be ~1e-15 with the
azimuth average nphi = 4); the helicity-forbidden scalar piece of psi_2 (exactly 0); scalar-diquark-only
<x2> = <x3> ([ud] symmetric).  The scalar-diquark contribution is UV finite; the axial-vector one has a
logarithmic UV divergence (the delta_{mu nu} coupling of the 1+ diquark to the leading-twist operator,
Gamma ~ 1/k^2 with no log suppression), so f_N and the moments are quoted versus Lambda.

Usage: python scripts/16_nucleon_da.py  -> results/nucleon_da.json
"""
import sys, json, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qdq.faddeev import NucleonFaddeev
from qdq.current import NucleonCurrent
from qdq.nucleon_da import NucleonDA

LAMS = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, None]


def run(D, B, tag):
    t = time.time()
    r = D.compute(B)
    tot0 = r['total'][0]
    row = dict(tag=tag, fN=abs(r['fN']), x1=r['x1'].real, x2=r['x2'].real, x3=r['x3'].real,
               resid=float(r['resid']), imag=float(max(abs(r['x1'].imag), abs(r['x2'].imag))),
               parts={k: (v / tot0).real.tolist() for k, v in r['parts'].items()},
               scalar={k: float(np.real(v)) for k, v in r['scalar'].items()},
               axial={k: float(np.real(v)) for k, v in r['axial'].items()}, sec=time.time() - t)
    print(f"{tag:26s} f_N = {row['fN']*1e3:6.3f}e-3 GeV^2   <x1> {row['x1']:.4f}  <x2> {row['x2']:.4f}  "
          f"<x3> {row['x3']:.4f}   scalar share {row['scalar']['share']:.3f}  resid {row['resid']:.0e}  "
          f"({row['sec']:.0f}s)", flush=True)
    return row


def main():
    t0 = time.time()
    F = NucleonFaddeev(g=(14.94 / 2, 1j * 12.98 / 2), nr=16, nz=8, ny=8, ncheb=6)
    F.solve_mass(1.17, 1.19, verbose=False)
    J = NucleonCurrent(F, nk=48, nz=32, ny=24)
    B = J.normalisation()[0]
    print(f"M_N = {F.M:.5f} GeV, B = {B:.6g}  ({time.time()-t0:.0f}s)", flush=True)
    out = dict(M=F.M, rows=[])
    # structural checks (azimuth-averaged outer loop)
    D = NucleonDA(J, inner=(32, 10, 8, 8), outer=(16, 8, 6), nphi=4, kmax_in=5.0, nproc=4)
    out['check'] = run(D, B, 'check nphi=4, Lambda=5')
    # outer-grid convergence at fixed Lambda
    for og in ((16, 8, 6), (24, 12, 8), (32, 16, 10)):
        D = NucleonDA(J, inner=(64, 12, 10, 8), outer=og, kmax_in=5.0, nproc=4)
        out['rows'].append(dict(run(D, B, f'outer {og}, Lambda=5'), outer=og, Lam=5.0))
    # inner-grid convergence at fixed Lambda
    for ig in ((64, 16, 12, 10), (96, 12, 10, 8)):
        D = NucleonDA(J, inner=ig, outer=(24, 12, 8), kmax_in=5.0, nproc=4)
        out['rows'].append(dict(run(D, B, f'inner {ig}, Lambda=5'), inner=ig, Lam=5.0))
    # cutoff scan
    out['scan'] = []
    for L in LAMS:
        D = NucleonDA(J, inner=(64, 12, 10, 8), outer=(24, 12, 8), kmax_in=L, nproc=4)
        row = run(D, B, f'Lambda = {L}')
        row['Lam'] = L
        out['scan'].append(row)
    pathlib.Path('results').mkdir(exist_ok=True)
    json.dump(out, open('results/nucleon_da.json', 'w'), indent=1, default=str)
    print(f"saved results/nucleon_da.json  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
