"""Nucleon EM current, two-loop diagrams 4-6: photon coupling inside the Faddeev kernel.

Derived by gauging OUR Faddeev kernel (qdq/faddeev.py),
    K_XY(k, l) = -4 g_X g_Y c_XY  Gamma^Y(rel_in) S^T(q) Gammabar^X(rel_out),
    q = l_qq - k_q,  rel_in = (k_q - q)/2,  rel_out = (l_q - q')/2,  q' = q + Q = k_qq - l_q,
so that  J^{4-6}_mu = int_k int_l  Psibar_X(k;-P_f) G0_X(k) K^mu_XY(k,l) G0_Y(l) Psi_Y(l;P_i),
with G0 = S(quark) Delta(diquark) and insertions in the same convention as diagrams 1-3
(Q.V = G(in) - G(out), i.e. V -> -dG at Q -> 0):
  diagram 4: S^T(q) -> [i chi_mu(q', q)]^T                    charge on the exchanged quark
  diagram 5: Gamma^Y(rel) -> X^Y_mu (outgoing legs), exchanged quark then carries q'
             X_mu(l;Q) = e_by (4l-Q)_mu/(4l.Q-Q^2)[G(l-Q/2)-G(l)] + e_ex (4l+Q)_mu/(4l.Q+Q^2)[G(l+Q/2)-G(l)],
             l = (k_q - q')/2    (Oettel et al.; Cheng et al. 2507.13484 Eq. 33)
  diagram 6: Gammabar^X(rel) -> -Xbar^X_mu (incoming legs: opposite sign in this convention)
             Xbar_mu(l;Q) = e_by (4l+Q)_mu/(4l.Q+Q^2)[Gb(l+Q/2)-Gb(l)] + e_ex (4l-Q)_mu/(4l.Q-Q^2)[Gb(l-Q/2)-Gb(l)],
             l = (l_q - q)/2
Flavour factors (2611.19819 App. B): exchanged quark  phi^T s^mT t^n Q t^mT s^n phi,
bystander in diagram 5  phi^T s^mT Q t^n t^mT s^n phi,  diagram 6  phi^T s^mT t^n t^mT Q s^n phi.
All divided differences are evaluated without kinematic singularities.
"""
import numpy as np
from .dirac import I4
from .quark import S as Sq, calF
from .quadrature import radial, chebyshev_u
from .faddeev import dot, GAM_S, GAM_A, BAR_S, BAR_A, SF, TF, _GRP
from .current import divdiff, EQ, EB

PIECES = ['ex4', 'by5', 'ex5', 'by6', 'ex6']

def exchange_flavour(charge, phi):
    """Flavour x charge factor of each piece for each block XY (X = outgoing diquark type)."""
    out = {}
    for X in 'sa':
        for Y in 'sa':
            c = {}
            prs = [(m, n) for m in _GRP[X] for n in _GRP[Y]]
            c['ex'] = sum(phi @ SF[m].T @ TF[n] @ charge.T @ TF[m].T @ SF[n] @ phi for m, n in prs).real
            c['by5'] = sum(phi @ SF[m].T @ charge @ TF[n] @ TF[m].T @ SF[n] @ phi for m, n in prs).real
            c['by6'] = sum(phi @ SF[m].T @ TF[n] @ TF[m].T @ charge @ SF[n] @ phi for m, n in prs).real
            out[X + Y] = {'ex4': c['ex'], 'by5': c['by5'], 'ex5': c['ex'], 'by6': c['by6'], 'ex6': c['ex']}
    return out

class ExchangeCurrent:
    def __init__(self, cur, nk=(16, 8, 8), nl=(16, 8, 8, 8), scale=0.6):
        """cur: a NucleonCurrent (supplies amplitudes, vertices, projectors, normalisation).
        nk = (radial, z, y) for the outer momentum k; nl = (radial, z, y, phi) for the inner l."""
        self.cur, self.fad, self.V = cur, cur.fad, cur.V
        self.M, self.eta = cur.M, cur.eta
        fad = self.fad
        self.g = {'s': fad.gs, 'a': fad.ga}
        self.w2 = {'s': fad.ws2, 'a': fad.wa2}
        # outer grid (azimuth fixed, factor 2 pi)
        r, wr = radial(nk[0], scale); z, wz = chebyshev_u(nk[1]); y, wy = np.polynomial.legendre.leggauss(nk[2])
        R, Z, Y = np.meshgrid(r, z, y, indexing='ij')
        sz, sy = np.sqrt(1 - Z**2), np.sqrt(1 - Y**2)
        self.k = np.stack([R * sz * sy, 0 * R, R * sz * Y, R * Z], -1).reshape(-1, 4)
        self.wk = (wr[:, None, None] * r[:, None, None]**3 * wz[None, :, None] * wy[None, None, :]
                   * 2 * np.pi / (2 * np.pi)**4).reshape(-1)
        # inner grid with explicit relative azimuth phi
        r, wr = radial(nl[0], scale); z, wz = chebyshev_u(nl[1]); y, wy = np.polynomial.legendre.leggauss(nl[2])
        nph = nl[3]
        ph = 2 * np.pi * (np.arange(nph) + 0.5) / nph
        R, Z, Y, PH = np.meshgrid(r, z, y, ph, indexing='ij')
        sz, sy = np.sqrt(1 - Z**2), np.sqrt(1 - Y**2)
        self.l = np.stack([R * sz * sy * np.cos(PH), R * sz * sy * np.sin(PH), R * sz * Y, R * Z], -1).reshape(-1, 4)
        self.wl = (wr[:, None, None, None] * r[:, None, None, None]**3 * wz[None, :, None, None]
                   * wy[None, None, :, None] * np.full(nph, 2 * np.pi / nph) / (2 * np.pi)**4).reshape(-1)

    def F(self, T, x):
        return calF(x / self.w2[T])

    def _outer_inner(self, Pf, Pi):
        """G0-dressed conjugate amplitude at the outer points and amplitude at the inner points."""
        cur, V, eta = self.cur, self.V, self.eta
        k, l = self.k, self.l
        Cs, Ca = cur.conj_amplitude(k, Pf)
        kq, kqq = eta * Pf + k, (1 - eta) * Pf - k
        Sk = Sq(kq)
        out = {'s': (V.D0(dot(kqq, kqq))[:, None, None] * (Cs @ Sk))[:, None],          # (Nk,1,4,4)
               'a': np.einsum('nam,naxy,nyz->nmxz', V.D1mat(kqq), Ca, Sk)}              # (Nk,mu',4,4)
        Ps, Pa = cur.amplitude(l, Pi)
        lq, lqq = eta * Pi + l, (1 - eta) * Pi - l
        Sl = Sq(lq)
        inn = {'s': (V.D0(dot(lqq, lqq))[:, None, None] * (Sl @ Ps))[:, None],          # (Nl,1,4,4)
               'a': np.einsum('nxy,nrv,nvyz->nrxz', Sl, V.D1mat(lqq), Pa)}              # (Nl,rho,4,4)
        return out, inn, kq, kqq, lq, lqq

    def compute(self, Q, photon=True, deriv=False, h=1e-4, probe=None):
        """Raw (flavour-stripped) projected contributions  res[XY][piece] -> (nb,) array.
        probe: EMProbe (default) or GravityProbe (see qdq/probes.py).
        photon=False: plain  Psibar G0 K G0 Psi  at Q=0 (trace / 2), piece 'K'.
        deriv=True:   -Psibar G0 dK/dP G0 Psi at Q=0 projected on T1, piece 'dK'."""
        cur, V = self.cur, self.V
        if photon and not deriv:
            Pf, Pi, Qv = cur.kinematics(Q)
        else:
            Pf = Pi = np.array([0, 0, 0, 1j * self.M]); Qv = np.zeros(4, complex)
        if probe is None:
            from .probes import EMProbe
            probe = EMProbe((self.fad.ms, self.fad.ma), self.M, self.V.p)
        if photon and not deriv:
            Tt, Ginv = probe.projectors(self.M, Pf, Pi, Qv)
        else:
            Tt, Ginv = cur.projectors(Pf, Pi, Qv)
        if not photon:
            Tt = I4[None, None]                       # (b=1, mu=1, 4, 4): plain trace
        if deriv:
            Tt = Tt[:1]
        out, inn, kq_all, kqq_all, lq, lqq = self._outer_inner(Pf, Pi)
        gam = {'s': GAM_S[None], 'a': GAM_A}
        bar = {'s': BAR_S[None], 'a': BAR_A}
        blocks = [X + Y for X in 'sa' for Y in 'sa']
        # inner-only products Bp[XY] = Gammabar^X_mu' B^Y_rho : (Nl, mu', rho, 4, 4) -> matrix
        Nl = len(self.wl)
        Bp = {}
        for XY in blocks:
            X, Y = XY
            B = np.einsum('mab,nrbc->nmrac', bar[X], inn[Y])
            Bp[XY] = B.transpose(0, 3, 1, 2, 4).reshape(Nl * 4, -1)                      # rows (n,a)
        res = {XY: {} for XY in blocks}
        acc = {XY: {} for XY in blocks}
        for ik in range(len(self.wk)):
            k = self.k[ik]
            kq, kqq = kq_all[ik], kqq_all[ik]
            q = lqq - kq                                       # exchanged quark, before photon
            qp = kqq - lq                                      # after photon (= q + Q)
            rel_in, rel_out = (kq - q) / 2, (lq - qp) / 2
            mids = {}
            if deriv:
                def kern(Ps):
                    lqq_s = (1 - self.eta) * Ps - self.l
                    lq_s = self.eta * Ps + self.l
                    kq_s, kqq_s = self.eta * Ps + k, (1 - self.eta) * Ps - k
                    qs = lqq_s - kq_s
                    ri, ro = (kq_s - qs) / 2, (lq_s - (kqq_s - lq_s)) / 2
                    return qs, ri, ro
                dS = []
                for mu in range(4):
                    e = np.zeros(4, complex); e[mu] = h
                    vals = []
                    for sgn in (1, -1):
                        qs, ri, ro = kern(Pf + sgn * e)
                        vals.append((qs, ri, ro))
                    dS.append(vals)
            for XY in blocks:
                X, Y = XY
                A = np.einsum('bmxy,pyz->bmpxz', Tt, out[X][ik])                          # (b,mu,mu',4,4)
                # Ap[(mu',rho,x),(b,mu,c)] = (A_{b mu mu'} Gamma^Y_rho)[x,c]
                Ap = np.einsum('bmpxy,ryz->prxbmz', A, gam[Y])
                Ap = Ap.reshape(-1, Ap.shape[3] * Ap.shape[4] * 4)
                W = (Bp[XY] @ Ap).reshape(Nl, 4, A.shape[0], A.shape[1], 4)              # [n, a, b, mu, c]
                pref = -4 * self.g[X] * self.g[Y]
                if not photon:
                    mid = (self.F(Y, dot(rel_in, rel_in)) * self.F(X, dot(rel_out, rel_out)))[:, None, None, None] \
                          * np.swapaxes(Sq(q), -1, -2)[:, None]
                    acc[XY].setdefault('K', 0)
                    acc[XY]['K'] = acc[XY]['K'] + self.wk[ik] * pref * np.einsum('n,nabmc,nmca->b', self.wl, W, mid)
                    continue
                if deriv:
                    md = []
                    for mu in range(4):
                        ms = []
                        for (qs, ri, ro) in dS[mu]:
                            ms.append((self.F(Y, dot(ri, ri)) * self.F(X, dot(ro, ro)))[:, None, None]
                                      * np.swapaxes(Sq(qs), -1, -2))
                        md.append(-(ms[0] - ms[1]) / (2 * h))
                    mid = np.stack(md, 1)                                                   # (Nl,mu,4,4)
                    acc[XY].setdefault('dK', 0)
                    acc[XY]['dK'] = acc[XY]['dK'] + self.wk[ik] * pref * np.einsum('n,nabmc,nmca->b', self.wl, W, mid)
                    continue
                FYin = self.F(Y, dot(rel_in, rel_in))
                FXout = self.F(X, dot(rel_out, rel_out))
                if 'chiT' not in mids:
                    mids['chiT'] = np.swapaxes(probe.quark(qp, q), -1, -2)                # (Nl,m,4,4)
                    mids['SqT'] = np.swapaxes(Sq(q), -1, -2)
                    mids['SqpT'] = np.swapaxes(Sq(qp), -1, -2)
                l5 = (kq - qp) / 2
                l6 = (lq - q) / 2
                FY = lambda x: self.F(Y, x)
                FX = lambda x: self.F(X, x)
                l5m, l5p = l5 - Qv / 2, l5 + Qv / 2
                l6m, l6p = l6 - Qv / 2, l6 + Qv / 2
                d5m = divdiff(FY, dot(l5m, l5m), dot(l5, l5))
                d5p = divdiff(FY, dot(l5p, l5p), dot(l5, l5))
                d6p = divdiff(FX, dot(l6p, l6p), dot(l6, l6))
                d6m = divdiff(FX, dot(l6m, l6m), dot(l6, l6))
                # seagulls: EM structure; the graviton lifts them with the leg momentum (bystander
                # k_q / l_q, exchanged q' / q) and adds delta_{mu nu}/2 x the unshifted amplitude
                sm5 = (FY(dot(l5, l5)) * FXout)[:, None, None] * mids['SqpT']
                sm6 = (FYin * FX(dot(l6, l6)))[:, None, None] * mids['SqT']
                v = {
                    'ex4': (FYin * FXout)[:, None, None, None] * mids['chiT'],
                    'by5': probe.lift((-0.25 * (4 * l5 - Qv) * (d5m * FXout)[:, None])[:, :, None, None]
                                      * mids['SqpT'][:, None], kq, sm5),
                    'ex5': probe.lift((0.25 * (4 * l5 + Qv) * (d5p * FXout)[:, None])[:, :, None, None]
                                      * mids['SqpT'][:, None], qp, sm5),
                    'by6': probe.lift((-0.25 * (4 * l6 + Qv) * (d6p * FYin)[:, None])[:, :, None, None]
                                      * mids['SqT'][:, None], lq, sm6),
                    'ex6': probe.lift((0.25 * (4 * l6 - Qv) * (d6m * FYin)[:, None])[:, :, None, None]
                                      * mids['SqT'][:, None], q, sm6),
                }
                for pc, mid in v.items():
                    acc[XY].setdefault(pc, 0)
                    acc[XY][pc] = acc[XY][pc] + self.wk[ik] * pref * np.einsum('n,nabmc,nmca->b', self.wl, W, mid)
        for XY in blocks:
            for pc, val in acc[XY].items():
                res[XY][pc] = val if not photon else Ginv @ val     # plain trace, or (F1,F2,F3)
        return res

    @staticmethod
    def combine(res, flav):
        """Sum raw block/piece contributions with flavour factors -> dict piece-group -> array."""
        tot = {'4': 0, '5': 0, '6': 0}
        for XY, d in res.items():
            for pc, val in d.items():
                tot[pc[-1]] = tot[pc[-1]] + flav[XY][pc] * val
        return tot
