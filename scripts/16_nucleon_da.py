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
    # outer UV cutoff (the Faddeev amplitude is frozen beyond ~113 GeV; the light-front weight grows
    # with |l|): grids (24,12,8) and (32,16,10) agree at fixed cutoff; the moments drift slowly with it
    for ko in (8.0, 15.0, 30.0):
        D = NucleonDA(J, inner=(64, 12, 10, 8), outer=(24, 12, 8), kmax_in=5.0, kmax_out=ko, nproc=4)
        out['rows'].append(dict(run(D, B, f'outer cut {ko}, Lambda=5'), kmax_out=ko, Lam=5.0))
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


# ------------------------------------------------------------------ analysis ----
# LO evolution of the first moments (Braun, Fries, Mahnke, Stein, hep-ph/0007279 Eq. 3.6; RQCD,
# arXiv:1512.02050 Eq. 2.23): relative to f_N the combinations
#   D_A = <x1> - <x3>  = 2 phi_10 / f_N  (P10 = 21(x1 - x3); gamma = 26/9 - 2/3 = 20/9),
#   D_B = <x1> - 2<x2> + <x3>  = 2 phi_11 / f_N  (P11 = 7(x1 - 2x2 + x3); gamma = 10/3 - 2/3 = 8/3),
# and f_N itself (gamma = 2/3), each scaling as exp(-gamma I), I = int alpha/(4 pi) dln zeta^2 in the
# normalisation of qdq/evolution.py (DGLAP: <x>_val ~ exp(-gamma_qq(1) I), gamma_qq(1) = 32/9).
# Comparison values at 2 GeV: RQCD 2019 (arXiv:1903.12590, Table 2: f_N = 3.54, phi_11 = 0.118,
# phi_10 = 0.182 in 1e-3 GeV^2; <x1> = 1/3 + phi10/fN + phi11/(3fN) etc.) and Mezrag et al. 2018
# (arXiv:1711.09101, Table I: f_N = 3.78(14), <x> = 0.379, 0.302, 0.319).

def evolve_moments(fN, x1, x2, x3, zeta=2.0):
    from qdq.evolution import AOEvolution, gamma_qq
    I = -np.log(AOEvolution().valence_factor(1, zeta)) / gamma_qq(1)
    DA, DB = (x1 - x3) * np.exp(-20 / 9 * I), (x1 - 2 * x2 + x3) * np.exp(-8 / 3 * I)
    d2 = -DB / 3
    d1, d3 = (DB / 3 + DA) / 2, (DB / 3 - DA) / 2
    return fN * np.exp(-2 / 3 * I), 1 / 3 + d1, 1 / 3 + d2, 1 / 3 + d3


def lattice_2019():
    fN, p11, p10 = 3.54, 0.118, 0.182
    return fN, 1 / 3 + p10 / fN + p11 / (3 * fN), 1 / 3 - 2 * p11 / (3 * fN), 1 / 3 - p10 / fN + p11 / (3 * fN)


def analyse():
    d = json.load(open('results/nucleon_da.json'))
    print("cutoff scan (hadron scale zeta_H, and LO-evolved to 2 GeV):")
    print(" Lambda   f_N[1e-3]   <x1>    <x2>    <x3>   scalar-share |  2 GeV: f_N   <x1>    <x2>    <x3>")
    for r in d['scan']:
        e = evolve_moments(r['fN'] * 1e3, r['x1'], r['x2'], r['x3'])
        print(f" {str(r['Lam']):>6s}   {r['fN']*1e3:7.3f}   {r['x1']:.4f}  {r['x2']:.4f}  {r['x3']:.4f}   {r['scalar']['share']:.3f}"
              f"      |        {e[0]:6.3f}  {e[1]:.4f}  {e[2]:.4f}  {e[3]:.4f}")
    print(" scalar-diquark part only:", [(r['Lam'], round(r['scalar']['x1'], 4), round(r['scalar']['x2'], 4),
                                          round(r['scalar']['x3'], 4)) for r in d['scan']])
    L = lattice_2019()
    print(f" RQCD 2019 (2 GeV):      f_N {L[0]:.2f}  <x> {L[1]:.4f} {L[2]:.4f} {L[3]:.4f}")
    print(" Mezrag et al. 2018 (2 GeV): f_N 3.78(14)  <x> 0.379 0.302 0.319")


if __name__ == '__main__':
    if '--analyse' in sys.argv:
        analyse()
    else:
        main()
