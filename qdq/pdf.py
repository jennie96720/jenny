"""Hadron-scale valence-quark distribution functions (DFs) of the nucleon: Mellin moments.

Following the quark+diquark DF construction of Yu, Cheng, Xing, (Gao), Roberts et al.
(EPJC 84 (2024) 739; arXiv:2505.10663) but with the momentum-dependent Faddeev amplitude, for which
the light-front delta function cannot be integrated analytically.  Instead we compute moments

    <x^m>_q = Tr[ n.J^(m)_q ] / Tr[ n.J^B ],
    n.J^(m)_q = sum over all insertions of the forward (Q = 0) number current of flavour q,
                each weighted by (n.p_struck / n.P)^m,

with a light-like n (n^2 = 0; Euclidean n = (0,0,1,i)) and the canonical normalisation
J^B_mu = (P_mu/M) (dlambda/dM) Psibar G0 Psi Lambda_+ (unit baryon number).
At Q = 0 every insertion is "-n.d/dp_struck" acting on the corresponding line of G0 K G0 (forward
Ward identity), so the weight is the light-front momentum of the struck line:
    diagram 1: bystander quark k_q;   diagram 4: exchanged quark q;
    diagram 5/6 seagulls: the leg whose momentum the amplitude's derivative acts on (k_q / q' and l_q / q);
    diagram 2: the DIQUARK k_d  -> gives the diquark light-front distribution <y^m>_D, which must be
               convolved with the quark-in-diquark DF:  <x^m> = <y^m>_D <z^m>_{q/D}  (step 2).
Diagram 3 (0+ <-> 1+ transition) vanishes at Q = 0 for unpolarised DFs.

Checks: m = 0 gives the u and d numbers (2, 1); m = 1 with <z>_{q/D} = 1/2 reproduces the
gravitational A^q(0); results are real and independent of the frame (boosted on-shell P).

UV cutoff kmax: the light-front weight grows like |k|^m.  In the exact integral the angular
integration suppresses it as (P/k)^m, but numerically any error that does not fall off is amplified
(beyond the largest Faddeev radial node the interpolated amplitude is frozen).  Integration points with
|k| > kmax (default 15 GeV) are therefore dropped; m <= 6 moments are stable for kmax in 8-20 GeV.
"""
import copy
import numpy as np
from .dirac import G, I4
from .quark import S as Sq
from .faddeev import dot, GAM_S, GAM_A, BAR_S, BAR_A, flavour_coeffs
from .current import flavour_factors, divdiff, lam_plus
from .exchange import exchange_flavour

NV = np.array([0, 0, 1, 1j])                       # light-like: n.n = 0

class NucleonPDFMoments:
    def __init__(self, cur, ex, mmax=8, n=NV, kmax=15.0):
        self.cur, self.fad, self.V = cur, cur.fad, cur.V
        self.M, self.eta = cur.M, cur.eta
        self.mmax, self.n = mmax, np.asarray(n, complex)
        self.ms = np.arange(mmax + 1)
        keep = np.linalg.norm(cur.k.real, axis=1) < kmax                   # one-loop grid
        self.k1, self.w1 = cur.k[keep], cur.w[keep]
        self.ex = None
        if ex is not None:                                                 # two-loop grids
            self.ex = copy.copy(ex)
            ko = np.linalg.norm(ex.k.real, axis=1) < kmax
            ki = np.linalg.norm(ex.l.real, axis=1) < kmax
            self.ex.k, self.ex.wk = ex.k[ko], ex.wk[ko]
            self.ex.l, self.ex.wl = ex.l[ki], ex.wl[ki]

    def _w(self, p, P):
        """(n.p / n.P)^m for m = 0..mmax, p: (N,4) -> (mmax+1, N)."""
        x = (p @ self.n) / (self.n @ P)
        return x[None, :] ** self.ms[:, None]

    def norm(self, P, dM=1e-3):
        """Tr[n.J^B] for unit baryon number (eigenvalue formula), at on-shell P (P^2 = -M^2)."""
        lp = self.fad.eigen(self.M + dM)[0]
        lm = self.fad.eigen(self.M - dM)[0]
        dlam = (lp - lm) / (2 * dM)
        n0 = self.psibar_G0_psi(P)
        return (self.n @ P) / self.M * dlam * n0 * 2, dlam, n0

    def psibar_G0_psi(self, P):
        """n0 with int Psibar G0 Psi = n0 Lambda_+(P)."""
        cur, V, eta = self.cur, self.V, self.eta
        k, w = self.k1, self.w1
        Ps, Pa = cur.amplitude(k, P)
        Cs, Ca = cur.conj_amplitude(k, P)
        kq, kd = eta * P + k, (1 - eta) * P - k
        S = Sq(kq)
        js = Cs @ S @ Ps * V.D0(dot(kd, kd))[:, None, None]
        ja = np.einsum('naxy,nyz,nab,nbzw->nxw', Ca, S, V.D1mat(kd), Pa)
        return (np.einsum('nxx->n', js + ja) @ w) / 2

    # ------------------------------------------------------------- one loop ----
    def one_loop(self, P, charge, phi=np.array([1., 0.])):
        """Weighted traces Tr[n.J^(m)] of diagrams 1 (quark) and 2 (diquark), split by channel:
        returns dict {'q_s','q_a','D_s','D_a'} -> (mmax+1,) arrays (flavour factors included)."""
        cur, V, eta, n = self.cur, self.V, self.eta, self.n
        k, w = self.k1, self.w1
        f1, f2, f3 = flavour_factors(charge, phi)
        Ps, Pa = cur.amplitude(k, P)
        Cs, Ca = cur.conj_amplitude(k, P)
        kq, kd = eta * P + k, (1 - eta) * P - k
        # diagram 1: i chi_mu(k_q,k_q) = -dS/dk_mu, contracted with n
        vq = 1j * np.einsum('m,nmab->nab', n, V.chi_quark(kq, kq))
        D1k = V.D1mat(kd)
        ts = f1['s'] * np.einsum('nxy,nyz,nzx->n', Cs, vq, Ps) * V.D0(dot(kd, kd))
        Ya = np.einsum('nab,nbcd->nacd', D1k, Pa)
        ta = f1['a'] * np.einsum('naxy,nyz,nazx->n', Ca, vq, Ya)
        wq = self._w(kq, P)
        out = {'q_s': wq @ (w * ts), 'q_a': wq @ (w * ta)}
        # diagram 2: diquark vertex at Q = 0, contracted with n; weight = diquark momentum
        SPs = Sq(kq) @ Ps
        SPa = Sq(kq)[:, None] @ Pa
        c0 = V.chi_s(kd, kd) @ n
        ts = f2['s'] * c0 * np.einsum('nxy,nyx->n', Cs, SPs)
        c1 = np.einsum('m,nmab->nab', n, V.chi_a(kd, kd))
        ta = f2['a'] * np.einsum('naxy,nab,nbyx->n', Ca, c1, SPa)
        wd = self._w(kd, P)
        out['D_s'] = wd @ (w * ts)
        out['D_a'] = wd @ (w * ta)
        return out

    # ------------------------------------------------------------- two loop ----
    def two_loop(self, P):
        """Flavour-stripped weighted traces of diagrams 4-6: res[XY][piece] -> (mmax+1,)."""
        ex, V, eta, n = self.ex, self.V, self.eta, self.n
        out, inn, kq_all, kqq_all, lq, lqq = ex._outer_inner(P, P)
        gam = {'s': GAM_S[None], 'a': GAM_A}
        bar = {'s': BAR_S[None], 'a': BAR_A}
        blocks = [X + Y for X in 'sa' for Y in 'sa']
        Nl = len(ex.wl)
        Tt = n[None, :, None, None] * I4                       # (b=1, mu, 4, 4): n_mu x identity
        Bp = {}
        for XY in blocks:
            X, Y = XY
            B = np.einsum('mab,nrbc->nmrac', bar[X], inn[Y])
            Bp[XY] = B.transpose(0, 3, 1, 2, 4).reshape(Nl * 4, -1)
        acc = {XY: {pc: 0 for pc in ['ex4', 'by5', 'ex5', 'by6', 'ex6']} for XY in blocks}
        nP = n @ P
        for ik in range(len(ex.wk)):
            kq, kqq = kq_all[ik], kqq_all[ik]
            q = lqq - kq                       # exchanged quark (Q = 0: q' = q)
            qp = kqq - lq
            rel_in, rel_out = (kq - q) / 2, (lq - qp) / 2
            l5, l6 = (kq - qp) / 2, (lq - q) / 2
            chiT = np.swapaxes(1j * V.chi_quark(qp, q), -1, -2)
            SqT, SqpT = np.swapaxes(Sq(q), -1, -2), np.swapaxes(Sq(qp), -1, -2)
            wts = {'ex4': (q @ n) / nP, 'by5': np.full(Nl, (kq @ n) / nP), 'ex5': (qp @ n) / nP,
                   'by6': (lq @ n) / nP, 'ex6': (q @ n) / nP}
            for XY in blocks:
                X, Y = XY
                A = np.einsum('bmxy,pyz->bmpxz', Tt, out[X][ik])
                Ap = np.einsum('bmpxy,ryz->prxbmz', A, gam[Y])
                Ap = Ap.reshape(-1, Ap.shape[3] * Ap.shape[4] * 4)
                W = (Bp[XY] @ Ap).reshape(Nl, 4, 1, 4, 4)
                pref = -4 * ex.g[X] * ex.g[Y] * ex.wk[ik]
                FYin, FXout = ex.F(Y, dot(rel_in, rel_in)), ex.F(X, dot(rel_out, rel_out))
                FY = lambda x: ex.F(Y, x)
                FX = lambda x: ex.F(X, x)
                d5 = divdiff(FY, dot(l5, l5), dot(l5, l5))         # Q = 0: derivatives
                d6 = divdiff(FX, dot(l6, l6), dot(l6, l6))
                mids = {
                    'ex4': (FYin * FXout)[:, None, None, None] * chiT,
                    'by5': (-l5 * (d5 * FXout)[:, None])[:, :, None, None] * SqpT[:, None],
                    'ex5': (l5 * (d5 * FXout)[:, None])[:, :, None, None] * SqpT[:, None],
                    'by6': (-l6 * (d6 * FYin)[:, None])[:, :, None, None] * SqT[:, None],
                    'ex6': (l6 * (d6 * FYin)[:, None])[:, :, None, None] * SqT[:, None],
                }
                for pc, mid in mids.items():
                    t = np.einsum('nabmc,nmca->n', W, mid)        # (Nl,)
                    wm = wts[pc][None, :] ** self.ms[:, None]
                    acc[XY][pc] = acc[XY][pc] + pref * (wm @ (ex.wl * t))
        return acc

    # ----------------------------------------------------------- assemble ----
    def moments(self, P, z_moments=None, two=None):
        """<x^m>_u, <x^m>_d (proton) and their diagram breakdown at on-shell P.
        z_moments: {'s': (mmax+1,), 'a': (mmax+1,)} quark-in-diquark DF moments (default: <z^m> = 2^-m,
        i.e. only the m = 0, 1 moments are meaningful)."""
        if z_moments is None:
            z = 0.5 ** self.ms
            z_moments = {'s': z, 'a': z}
        NB, dlam, n0 = self.norm(P)
        if two is None:
            two = self.two_loop(P)
        res = {}
        for fl, tau in (('u', np.diag([1., 0.])), ('d', np.diag([0., 1.]))):
            o = self.one_loop(P, tau)
            cx = exchange_flavour(tau, np.array([1., 0.]))
            d = {'1': o['q_s'] + o['q_a'],
                 '2': o['D_s'] * z_moments['s'] + o['D_a'] * z_moments['a'],
                 '4': 0, '5': 0, '6': 0}
            for XY, pcs in two.items():
                for pc, val in pcs.items():
                    d[pc[-1]] = d[pc[-1]] + cx[XY][pc] * val
            res[fl] = {k: v / NB for k, v in d.items()}
            res[fl]['total'] = sum(res[fl][k] for k in '12456')
            res[fl]['diquark_y'] = {'s': o['D_s'] / NB, 'a': o['D_a'] / NB}
        return res
