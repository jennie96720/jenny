"""Leading-twist (twist-3) distribution amplitude of the proton from the quark+diquark Faddeev
amplitude: normalisation f_N and low Mellin moments.

Definition (Mezrag, Segovia, Chang, Roberts, arXiv:1711.09101, Eq. 1; Euclidean metric):
    <0| eps^{abc} (u^a_+)^T C^dag n.gamma u^b_-  n.gamma d^c_+ |P>
        = (1/2) f_N n.P  n.gamma N_+  int [dx] phi(x1,x2,x3) e^{...},     q_pm = H_pm q, H_pm = (1 pm g5)/2,
with int[dx] phi = 1.  Mellin moments  <x1^l x2^m> = int [dx] x1^l x2^m phi,  x_i = n.p_i / n.P.

Three-quark Bethe-Salpeter wave function (unamputated), Psi = psi_1 + psi_2 + psi_3 (bystander label),
psi_2, psi_1 from psi_3 by cyclic relabelling 1->2->3->1:
    psi_3 = [S(p1) Gamma^D(k) S^T(p2)]_{a1 a2} Delta^D(K) [S(p3) Psi_D(l;P)]_{a3 delta},
    p3 = l + P/3,  K = P - p3 = -l + 2P/3,  p1 = k + K/2,  p2 = -k + K/2,
with the conventions of the Faddeev kernel (qdq/faddeev.py): Gamma^0+ = g C i g5 F(k^2/w^2),
Gamma^1+_mu = g i g_mu C F(k^2/w^2), Delta of qdq/faddeev.py, Psi_D the canonically normalised
quark-diquark amplitude (Lambda_+ included).  Colour: three-quark colour wave function eps/sqrt3 (the
diquark colour trace 2 sits in g), contracted with eps^{abc}: factor 6/sqrt3 for every component.
Flavour (u,u,d) in slots (1,2,3), from t^j, s^j phi of qdq/faddeev.py:
    psi_3: {uu}d  sqrt(2/3) (axial only);
    psi_1: [ud] 1/sqrt2,  {ud} -1/sqrt6   (diquark = slots (2,3));
    psi_2: [du] -1/sqrt2, {du} -1/sqrt6   (diquark = slots (3,1)).
Dirac contraction with O = H_+^T C^dag n.gamma H_-  (slots 1,2) and n.gamma H_+ (slot 3):
    psi_3:  Tr[O^T M_3] n.gamma H_+ X3,     psi_1:  n.gamma H_+ M_1^T O^T X1,     psi_2:  n.gamma H_+ M_2 O X2,
    M = int_k S(p_a) Gamma S^T(p_b)  (diquark loop),  X = S(p_bystander) Psi_D  (Delta contracted in).
The result R must be proportional to n.gamma H_+ Lambda_+; R = (1/2) f_N n.P n.gamma H_+ Lambda_+ <w>.

Numerics: the diquark loop is done analytically in the Dirac algebra,
    S(pa) G S^T(pb) = sS_a sS_b G - i sV_a sS_b (pa.g) G - i sS_a sV_b G (pb.g)^T - sV_a sV_b (pa.g) G (pb.g)^T,
so only the scalar/vector/tensor integrals of sigma-products (times F and the light-front weight) are
needed for each outer point; the inner grid is a full 4D grid (|k|, z, y, cos phi): K and n span three
Euclidean directions.  The outer integrand depends on l only through l^2, l.P, n.l, so the azimuth about
axis 3 is trivial for any rotation-invariant projection (the projection onto n.gamma H_+ Lambda_+ is).
"""
import numpy as np
from .dirac import G, G5, C, Cdag, I4, slash
from .quark import S as Sq, sigmas, calF
from .quadrature import radial, chebyshev_u
from .faddeev import dot

NV = np.array([0, 0, 1, 1j])
HP, HM = (I4 + G5) / 2, (I4 - G5) / 2
GT = np.swapaxes(G, -1, -2)                       # gamma_mu^T
GAM = {'s': (C @ (1j * G5))[None], 'a': np.array([1j * G[m] @ C for m in range(4)])}


def inner_grid(nq=24, nz=12, ny=10, nphi=8, scale=0.6, kmax=None):
    """Full 4D grid: k = |k| (sz sy cos phi, sz sy sin phi, sz y, z), measure d^4k/(2pi)^4.
    phi: nphi uniform nodes on (0, 2pi), symmetric under phi -> -phi (trapezoid rule, spectrally
    accurate for the periodic integrand).  The full circle is needed: the Dirac structure carries
    terms odd in k_2 (gamma_2 k_2), which cancel only between k_2 and -k_2."""
    q, wq = radial(nq, scale)
    if kmax is not None:
        keep = q < kmax
        q, wq = q[keep], wq[keep]
    z, wz = chebyshev_u(nz)
    y, wy = np.polynomial.legendre.leggauss(ny)
    ph = 2 * np.pi * (np.arange(nphi) + 0.5) / nphi
    wp = np.full(nphi, 2 * np.pi / nphi)
    Q, Z, Y, PH = np.meshgrid(q, z, y, ph, indexing='ij')
    sz, sy = np.sqrt(1 - Z**2), np.sqrt(1 - Y**2)
    k = np.stack([Q * sz * sy * np.cos(PH), Q * sz * sy * np.sin(PH), Q * sz * Y, Q * Z], -1)
    w = (wq[:, None, None, None] * q[:, None, None, None]**3 * wz[None, :, None, None]
         * wy[None, None, :, None] * wp[None, None, None, :]) / (2 * np.pi)**4
    return k.reshape(-1, 4), w.reshape(-1)


def outer_grid(nl=24, nz=12, ny=10, nphi=1, scale=0.6, kmax=None):
    """l = |l| (sz sy cos phi, sz sy sin phi, sz y, z); nphi = 1: the trivial azimuth (factor 2 pi)."""
    r, wr = radial(nl, scale)
    if kmax is not None:
        keep = r < kmax
        r, wr = r[keep], wr[keep]
    z, wz = chebyshev_u(nz)
    y, wy = np.polynomial.legendre.leggauss(ny)
    ph = 2 * np.pi * (np.arange(nphi) + 0.5) / nphi
    R, Z, Y, PH = np.meshgrid(r, z, y, ph, indexing='ij')
    sz, sy = np.sqrt(1 - Z**2), np.sqrt(1 - Y**2)
    l = np.stack([R * sz * sy * np.cos(PH), R * sz * sy * np.sin(PH), R * sz * Y, R * Z], -1)
    w = (wr[:, None, None, None] * r[:, None, None, None]**3 * wz[None, :, None, None]
         * wy[None, None, :, None] * (2 * np.pi / nphi)) / (2 * np.pi)**4 * np.ones_like(PH)
    return l.reshape(-1, 4), w.reshape(-1)


class NucleonDA:
    """f_N and <1>, <x1>, <x2> of the proton's leading-twist DA (unit-normalised Faddeev amplitude)."""

    def __init__(self, cur, inner=(24, 12, 10, 8), outer=(24, 12, 10), nphi=1, scale=0.6,
                 kmax_in=None, kmax_out=None, n=NV, chunk=64, nproc=1):
        self.cur, self.fad = cur, cur.fad
        self.nproc = nproc
        self.M = cur.M
        self.P = np.array([0, 0, 0, 1j * self.M])
        self.n = np.asarray(n, complex)
        self.nP = self.n @ self.P
        self.k, self.wk = inner_grid(*inner, scale=scale, kmax=kmax_in)
        self.l, self.wl = outer_grid(*outer, nphi=nphi, scale=scale, kmax=kmax_out)
        self.chunk = chunk
        self.mass = {'s': self.fad.ms, 'a': self.fad.ma}
        self.g = {'s': self.fad.gs, 'a': self.fad.ga}
        self.w2 = {c: 0.5 * m**2 for c, m in self.mass.items()}
        self.Fin = {c: calF(dot(self.k, self.k) / self.w2[c]) for c in 'sa'}
        X = slash(self.n) @ HP
        self.Xproj = X @ self._lam_plus()
        self.O = HP.T @ Cdag @ slash(self.n) @ HM

    def _lam_plus(self):
        return (-1j * slash(self.P) + self.M * I4) / (2 * self.M)

    # ---------------------------------------------------------------- diquark loop ----
    def _inner(self, K):
        """For outer diquark momenta K (N,4): sigma-product integrals with weights 1, x_a, x_b.
        Returns dict c -> (A0 (3,N), Aa (3,N,4), Ab (3,N,4), Aab (3,N,4,4)) for c = 's','a'."""
        out = {c: [np.zeros((3, len(K)), complex), np.zeros((3, len(K), 4), complex),
                   np.zeros((3, len(K), 4), complex), np.zeros((3, len(K), 4, 4), complex)] for c in 'sa'}
        k, wk = self.k, self.wk
        for i0 in range(0, len(K), self.chunk):
            Kc = K[i0:i0 + self.chunk]
            pa = k[None] + Kc[:, None] / 2                     # (c, Ni, 4)
            pb = -k[None] + Kc[:, None] / 2
            sVa, sSa = sigmas(np.einsum('cim,cim->ci', pa, pa))
            sVb, sSb = sigmas(np.einsum('cim,cim->ci', pb, pb))
            xa, xb = (pa @ self.n) / self.nP, (pb @ self.n) / self.nP
            wts = np.stack([np.ones_like(xa), xa, xb])         # (3, c, Ni)
            for ch in 'sa':
                base = wts * (self.Fin[ch] * wk)[None, None]     # (3, c, Ni)
                A0, Aa, Ab, Aab = out[ch]
                A0[:, i0:i0 + len(Kc)] = np.einsum('wci,ci->wc', base, sSa * sSb)
                Aa[:, i0:i0 + len(Kc)] = np.einsum('wci,ci,cim->wcm', base, sVa * sSb, pa)
                Ab[:, i0:i0 + len(Kc)] = np.einsum('wci,ci,cim->wcm', base, sSa * sVb, pb)
                Aab[:, i0:i0 + len(Kc)] = np.einsum('wci,ci,cim,cin->wcmn', base, sVa * sVb, pa, pb,
                                                     optimize=True)
        return out

    def _inner_parallel(self, K):
        """_inner over outer points split across self.nproc forked workers (Linux)."""
        nproc = getattr(self, 'nproc', 1)
        if nproc <= 1 or len(K) < 4 * nproc:
            return self._inner(K)
        import multiprocessing as mp
        parts = np.array_split(K, nproc)
        global _DA_SELF
        _DA_SELF = self
        with mp.get_context('fork').Pool(nproc) as pool:
            res = pool.map(_inner_worker, parts)
        return {c: [np.concatenate([r[c][j] for r in res], axis=1) for j in range(4)] for c in 'sa'}

    def _M(self, A, ch):
        """int S(pa) Gamma S^T(pb) for Gamma of channel ch (g included): (3,N,[mu],4,4)."""
        A0, Aa, Ab, Aab = A
        Gm = self.g[ch] * GAM[ch]                               # (mu,4,4)
        t0 = np.einsum('wn,uab->wnuab', A0, Gm)
        t1 = -1j * np.einsum('wnm,mac,ucb->wnuab', Aa, G, Gm)
        t2 = -1j * np.einsum('wnm,uac,mcb->wnuab', Ab, Gm, GT)
        t3 = -np.einsum('wnmv,mac,ucd,vdb->wnuab', Aab, G, Gm, GT, optimize=True)
        M = t0 + t1 + t2 + t3
        return M[:, :, 0] if ch == 's' else M

    # ---------------------------------------------------------------- assembly ----
    def compute(self, B=None, split=False):
        """Returns dict with f_N, moments <x1>, <x2>, <x3>, per-component projections and the
        proportionality residual.  B: baryon-number normalisation of the raw amplitude
        (NucleonCurrent.normalisation()[0]); None -> raw amplitude."""
        P, n, M = self.P, self.n, self.M
        l, wl = self.l, self.wl
        pc = l + P / 3
        K = P - pc
        K2 = dot(K, K)
        Ps, Pa = self.cur.amplitude(l, P)                       # (N,4,4), (N,nu,4,4)
        Sc = Sq(pc)
        Xs = Sc @ Ps * (calF(K2 / self.w2['s']) / self.mass['s']**2)[:, None, None]
        Dlt = (np.eye(4)[None] + K[:, :, None] * K[:, None, :] / self.mass['a']**2) \
            * (calF(K2 / self.w2['a']) / self.mass['a']**2)[:, None, None]
        Xa = np.einsum('nuv,nvab->nuab', Dlt, Sc[:, None] @ Pa)  # (N,mu,4,4): Delta_{mu nu} S Psi_nu
        xc = (pc @ n) / self.nP
        A = self._inner_parallel(K)
        Ms, Ma = self._M(A['s'], 's'), self._M(A['a'], 'a')     # (3,N,4,4), (3,N,mu,4,4)
        NH = slash(n) @ HP
        O = self.O
        col = 6 / np.sqrt(3)
        fl = {'3a': np.sqrt(2 / 3), '1s': 1 / np.sqrt(2), '1a': -1 / np.sqrt(6),
              '2s': -1 / np.sqrt(2), '2a': -1 / np.sqrt(6)}
        R = {}
        # psi_3: diquark (1,2) = {uu}, bystander d (slot 3); inner weights x_a = x1, x_b = x2
        T = np.einsum('ab,wnuab->wnu', O, Ma)                    # sum_ab O_ab M_ab = Tr[O^T M_mu]
        r3 = np.einsum('wnu,xy,nuyz->wnxz', T, NH, Xa)
        R['3a'] = fl['3a'] * col * np.stack([r3[0], r3[1], r3[2]])
        # psi_1: diquark (2,3), bystander slot 1; x1 = x_c (outer), x2 = x_a (inner)
        def c1(Mw, Xw, ax):
            if ax:
                return np.einsum('xy,nuzy,az,nuab->nxb', NH, Mw, O, Xw)   # NH M^T O^T X
            return np.einsum('xy,nzy,az,nab->nxb', NH, Mw, O, Xw)
        for ch, Mx, Xx in (('s', Ms, Xs), ('a', Ma, Xa)):
            r = [c1(Mx[0], Xx, ch == 'a'), xc[:, None, None] * c1(Mx[0], Xx, ch == 'a'),
                 c1(Mx[1], Xx, ch == 'a')]
            R['1' + ch] = fl['1' + ch] * col * np.stack(r)
        # psi_2: diquark (3,1), bystander slot 2; x1 = x_b (inner), x2 = x_c (outer)
        def c2(Mw, Xw, ax):
            if ax:
                return np.einsum('xy,nuyz,za,nuab->nxb', NH, Mw, O, Xw)   # NH M O X
            return np.einsum('xy,nyz,za,nab->nxb', NH, Mw, O, Xw)
        for ch, Mx, Xx in (('s', Ms, Xs), ('a', Ma, Xa)):
            r = [c2(Mx[0], Xx, ch == 'a'), c2(Mx[2], Xx, ch == 'a'),
                 xc[:, None, None] * c2(Mx[0], Xx, ch == 'a')]
            R['2' + ch] = fl['2' + ch] * col * np.stack(r)
        norm = 1.0 if B is None else 1 / np.sqrt(B)
        Rint = {key: norm * np.einsum('n,wnab->wab', wl, v) for key, v in R.items()}  # (3,4,4)
        X = self.Xproj
        den = np.einsum('ab,ab->', X.conj(), X)
        proj = {key: np.einsum('ab,wab->w', X.conj(), v) / den for key, v in Rint.items()}
        tot = sum(proj.values())
        Rt = sum(Rint.values())
        resid = np.linalg.norm(Rt[0] - tot[0] * X) / np.linalg.norm(Rt[0])
        fN = 2 * tot[0] / self.nP
        res = dict(fN=fN, x1=tot[1] / tot[0], x2=tot[2] / tot[0],
                   x3=1 - (tot[1] + tot[2]) / tot[0], resid=resid,
                   parts={key: v for key, v in proj.items()}, total=tot)
        # scalar-diquark-only and axial-only splits (moments normalised within the subset)
        for name, keys in (('scalar', ('1s', '2s')), ('axial', ('3a', '1a', '2a'))):
            t = sum(proj[q] for q in keys)
            res[name] = dict(share=t[0] / tot[0], x1=t[1] / t[0], x2=t[2] / t[0],
                             x3=1 - (t[1] + t[2]) / t[0])
        return res


_DA_SELF = None


def _inner_worker(K):
    return _DA_SELF._inner(K)
