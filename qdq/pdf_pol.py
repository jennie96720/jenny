"""Helicity-dependent (polarised) valence-quark DFs of the proton: Mellin moments <x^m>_{Delta q}.

Same construction as qdq/pdf.py, with the forward (Q = 0) axial current gamma_5 gamma_mu of flavour q in
place of the number current, following the realistic quark+diquark axial current of Chen, Fischer,
Roberts, Segovia, arXiv:2103.02054 (PRD 105 (2022) 094022):
  quark vertex   Gamma_5mu(k,k) = gamma_5 [gamma_mu A(k^2) + 2 gamma.k k_mu A'(k^2)]   (Eq. 26 at Q = 0;
                 the pion-pole term ~ Q_mu/(Q^2 + m_pi^2) vanishes at Q = 0)
  diagram 1      probe on the bystander quark (scalar / axial-vector diquark spectator)
  diagram 2      probe on the axial-vector diquark:  Gamma^aa_{mu,al be}(K) = c_aa(K^2) eps_{mu al be nu} K_nu
  diagram 3      probe-induced 0+ <-> 1+ transition:  Gamma^sa_{mu,be} = c1(K^2) delta_{mu be} + c2(K^2) K_mu K_be/K^2
                 (and as); the scalar diquark has no axial coupling (parity)
  diagram 4      probe on the exchanged quark
  diagrams 5, 6  seagulls ~ Q_mu/(Q^2 + m_pi^2): zero at Q = 0.
The diquark couplings are computed here from the quark loop inside the diquark (axial_triangle),
with our own diquark amplitudes and gamma-matrix conventions, as functions of the diquark virtuality
K^2 (AxialCouplingTable): they fall steeply off-shell (c_aa = -1.11i on-shell, -0.18i at K^2 = 0), so
constant on-shell couplings overestimate diagrams 2, 3 by ~3x.  2103.02054 instead uses constants
(kappa_ax^aa = 0.73, kappa_ax^sa = 0.75) fixed by matching resolved nucleon-level diagrams.  Flavour factors: those of the vector
current with the flavour projector of quark q (2103.02054 App. C has the same structure).

Moments: <x^m>_{Delta q} = h_m / h_B, with h_m the coefficient of T = Lambda_+ gamma_5 gamma.n Lambda_+
in n.J_5^(m) (the struck-line weight (n.p/n.P)^m as in qdq/pdf.py) and h_B the coefficient of
Lambda_+ gamma.n Lambda_+ in the unit baryon-number current; m = 0 gives the axial charges, and
g_A = Delta u - Delta d (2103.02054: 1.25(3), g_A^d/g_A^u = -0.16(2)).  Diagrams 2 and 3 give diquark
LF distributions, convolved with quark-in-diquark DFs as in EPJC 84:739 Eqs. (20), (22):
<x^m> = <y^m>_D <z^m>_{q/D} (1+: phi_1^2; transition: phi_0 phi_1).
"""
import copy
import numpy as np
from .dirac import G, G5, I4, slash
from .quark import S as Sq, AB, calF
from .faddeev import dot, GAM_S, GAM_A, BAR_S, BAR_A
from .current import flavour_factors, divdiff, lam_plus, EPS
from .exchange import exchange_flavour
from .pdf import NucleonPDFMoments, NV

G5G = np.einsum('ab,mbc->mac', G5, G)                   # gamma_5 gamma_mu

def A_of(p2):
    return AB(p2)[0]

def axial_quark(l):
    """Unamputated forward axial vertex S(l) Gamma_5mu(l,l) S(l): l (N,4) -> (N, mu, 4, 4)."""
    p2 = dot(l, l)
    A = A_of(p2)
    dA = divdiff(A_of, p2, p2)
    V = A[:, None, None, None] * G5G[None] \
        + 2 * (G5 @ slash(l))[:, None] * (l * dA[:, None])[:, :, None, None]
    S = Sq(l)[:, None]
    return S @ V @ S

# ------------------------------------------------------ diquark axial couplings ----
def axial_triangle(X, Y, K, g, mdq=(0.79, 0.89), nq=48, nz=24, ny=16, nphi=8):
    """Axial current (Q = 0) on the quarks inside a diquark, quark loop as in current.diquark_triangle:
        T_a = int Tr[ Gbar^X(q; -K) Shat_mu(q_a) G^Y(q; K) S^T(q_b) ]
        T_b = int Tr[ Gbar^X(q; -K) S(q_a) G^Y(q; K) Shat_mu(q_b)^T ],   q_a = q + K/2, q_b = -q + K/2,
    Shat = unamputated axial vertex, colour trace 2, flavour stripped.  Returns T_a, T_b with indices
    (mu, i, j): i on Gbar^X, j on G^Y (scalar: single index 0).  Full 4D angular grid (the vertex has
    free Lorentz indices, so the azimuth is not trivial as it is for the traces in current.py)."""
    from .quadrature import radial, chebyshev_u
    w2 = {'s': 0.5 * mdq[0]**2, 'a': 0.5 * mdq[1]**2}
    gg = {'s': g[0], 'a': g[1]}
    r, wr = radial(nq, 1.0)
    z, wz = chebyshev_u(nz)
    y, wy = np.polynomial.legendre.leggauss(ny)
    phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    R, Z, Yy, Ph = np.meshgrid(r, z, y, phi, indexing='ij')
    sz, sy = np.sqrt(1 - Z**2), np.sqrt(1 - Yy**2)
    q = np.stack([R * sz * sy * np.cos(Ph), R * sz * sy * np.sin(Ph), R * sz * Yy, R * Z], -1).reshape(-1, 4)
    w = (wr[:, None, None, None] * r[:, None, None, None]**3 * wz[None, :, None, None]
         * wy[None, None, :, None] * np.full(nphi, 2 * np.pi / nphi)[None, None, None, :]
         / (2 * np.pi)**4).reshape(-1)
    qa, qb = q + K / 2, -q + K / 2
    def amp(T, bar):
        F = calF(dot(q, q) / w2[T]) * gg[T]
        D = (BAR_S[None] if T == 's' else BAR_A) if bar else (GAM_S[None] if T == 's' else GAM_A)
        return F[:, None, None, None] * D[None]
    Gx, Gy = amp(X, True), amp(Y, False)
    ins_a = axial_quark(qa)
    ins_b = np.swapaxes(axial_quark(qb), -1, -2)
    Ta = 2 * np.einsum('n,nixy,nmyz,njzw,nwx->mij', w, Gx, ins_a, Gy, np.swapaxes(Sq(qb), -1, -2))
    Tb = 2 * np.einsum('n,nixy,nyz,njzw,nmwx->mij', w, Gx, Sq(qa), Gy, ins_b)
    return Ta, Tb

def diquark_axial_couplings(g, mdq=(0.79, 0.89), **kw):
    """c_aa, c_sa, c_as of the amputated Q = 0 diquark axial vertices (per unit flavour weight of
    flavour_factors: elastic Tr[t^T(Qt + tQ)] -> -(T_a + T_b)/2, transition Tr[t^T(Qt - tQ)] ->
    -(T_a - T_b)/2, the sign as for the vector current in current.transition_kappas), projected on
    eps_{mu al be nu} K_nu (aa) and on the K-transverse delta_{mu be} (sa, as).  Also returns the
    scalar-diquark vertex (must vanish) and the T_a -/+ T_b asymmetries (must vanish)."""
    out = {}
    ma = mdq[1]
    K = np.array([0, 0, 0, 1j * ma])
    Ta, Tb = axial_triangle('a', 'a', K, g, mdq, **kw)
    V = -(Ta + Tb) / 2
    E = np.einsum('mabn,n->mab', EPS, K)
    out['aa'] = np.einsum('mab,mab->', V, E) / np.einsum('mab,mab->', E, E)
    out['aa_resid'] = np.abs(V - out['aa'] * E).max() / np.abs(V).max()
    out['aa_asym'] = np.abs(Ta - Tb).max() / np.abs(Ta).max()
    mbar = 0.5 * sum(mdq)
    K = np.array([0, 0, 0, 1j * mbar])
    PT = np.eye(4) - np.outer(K, K) / dot(K, K)
    for X, Y in (('s', 'a'), ('a', 's')):
        Ta, Tb = axial_triangle(X, Y, K, g, mdq, **kw)
        Ta, Tb = (Ta[:, 0, :], Tb[:, 0, :]) if X == 's' else (Ta[:, :, 0], Tb[:, :, 0])
        V = -(Ta - Tb) / 2
        out[X + Y] = np.einsum('mb,mb->', V, PT) / 3
        out[X + Y + '_asym'] = np.abs(Ta + Tb).max() / np.abs(Ta).max()
        out[X + Y + '_long'] = np.einsum('mb,m,b->', V, K, K) / dot(K, K)
    K = np.array([0, 0, 0, 1j * mdq[0]])
    Ta, Tb = axial_triangle('s', 's', K, g, mdq, **kw)
    out['ss_size'] = np.abs(Ta[:, 0, 0] + Tb[:, 0, 0]).max()
    return out

def axial_couplings_at(K2, g, mdq=(0.79, 0.89), **kw):
    """Off-shell couplings at complex K^2 (covariance: they depend on K^2 only):
    c_aa (eps.K structure), c1_sa / c2_sa (delta_{mu be} / K_mu K_be / K^2), same for 'as'."""
    kap = np.sqrt(-complex(K2))
    K = np.array([0, 0, 0, 1j * kap])
    if abs(K2) < 1e-8:
        K = np.array([0, 0, 1e-4, 0], complex)
    Ta, Tb = axial_triangle('a', 'a', K, g, mdq, **kw)
    V = -(Ta + Tb) / 2
    E = np.einsum('mabn,n->mab', EPS, K)
    out = {'aa': np.einsum('mab,mab->', V, E) / np.einsum('mab,mab->', E, E)}
    KK = np.outer(K, K) / dot(K, K)
    PT = np.eye(4) - KK
    for X, Y in (('s', 'a'), ('a', 's')):
        Ta, Tb = axial_triangle(X, Y, K, g, mdq, **kw)
        Ta, Tb = (Ta[:, 0, :], Tb[:, 0, :]) if X == 's' else (Ta[:, :, 0], Tb[:, :, 0])
        V = -(Ta - Tb) / 2
        c1 = np.einsum('mb,mb->', V, PT) / 3
        out['c1_' + X + Y] = c1
        out['c2_' + X + Y] = np.einsum('mb,mb->', V, KK) - c1      # V = c1 delta + c2 KK/K^2
    return out

class AxialCouplingTable:
    """c(K^2) on a grid in the upper half of the complex K^2 plane, bicubic interpolation of real and
    imaginary parts; lower half from c(z*) = -conj(c(z)) (c is i x a real-analytic function).
    Outside the grid: nearest edge (the couplings are small there)."""
    KEYS = ('aa', 'c1_sa', 'c2_sa', 'c1_as', 'c2_as')

    def __init__(self, re, im, values):
        from scipy.interpolate import RectBivariateSpline
        self.re, self.im = np.asarray(re), np.asarray(im)
        self.sp = {}
        for k in self.KEYS:
            v = np.asarray(values[k])
            self.sp[k] = (RectBivariateSpline(self.re, self.im, v.real, kx=3, ky=3),
                          RectBivariateSpline(self.re, self.im, v.imag, kx=3, ky=3))

    @classmethod
    def compute(cls, g, mdq=(0.79, 0.89), re=None, im=None, verbose=True, **kw):
        re = np.array([-0.62, -0.45, -0.3, -0.15, 0.0, 0.2, 0.45, 0.8, 1.3, 2.0, 3.0, 4.5]) if re is None else re
        im = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.2, 3.2]) if im is None else im
        vals = {k: np.zeros((len(re), len(im)), complex) for k in cls.KEYS}
        for i, a in enumerate(re):
            for j, b in enumerate(im):
                c = axial_couplings_at(complex(a, b), g, mdq, **kw)
                for k in cls.KEYS:
                    vals[k][i, j] = c[k]
            if verbose:
                print(f"  re K^2 = {a:5.2f} done", flush=True)
        return cls(re, im, vals), vals

    def __call__(self, key, K2):
        K2 = np.asarray(K2, complex)
        x = np.clip(K2.real, self.re[0], self.re[-1])
        y = np.abs(K2.imag)
        y = np.clip(y, self.im[0], self.im[-1])
        sr, si = self.sp[key]
        v = sr.ev(x, y) + 1j * si.ev(x, y)
        return np.where(K2.imag >= 0, v, -np.conj(v))

# ------------------------------------------------------------- moments ----
class NucleonHelicityMoments(NucleonPDFMoments):
    def __init__(self, cur, ex, couplings, mmax=8, n=NV, kmax=15.0, lams=(0.0,)):
        super().__init__(cur, ex, mmax=mmax, n=n, kmax=kmax, lams=lams)
        self.c = couplings

    def structures(self, P):
        L = lam_plus(P, self.M)
        T = L @ G5 @ slash(self.n) @ L
        TV = L @ slash(self.n) @ L
        X = T.conj().T
        return T, TV, X

    def one_loop(self, P, charge, phi=np.array([1., 0.])):
        """Weighted Tr[X n.J_5] of diagrams 1-3 by channel: {'q_s','q_a','D_aa','D_sa','D_as'} -> (nlam, m+1)."""
        cur, V, eta, n = self.cur, self.V, self.eta, self.n
        k, w = self.k1, self.w1
        T, TV, X = self.structures(P)
        f1, f2, f3 = flavour_factors(charge, phi)
        Ps, Pa = cur.amplitude(k, P)
        Cs, Ca = cur.conj_amplitude(k, P)
        kq, kd = eta * P + k, (1 - eta) * P - k
        vq = np.einsum('m,nmab->nab', n, axial_quark(kq))
        D0 = V.D0(dot(kd, kd))
        D1 = V.D1mat(kd)
        ts = f1['s'] * np.einsum('wx,nxy,nyz,nzw->n', X, Cs, vq, Ps) * D0
        Ya = np.einsum('nab,nbcd->nacd', D1, Pa)
        ta = f1['a'] * np.einsum('wx,naxy,nyz,nazw->n', X, Ca, vq, Ya)
        wq = self._w(kq, P)
        out = {'q_s': wq @ (w * ts), 'q_a': wq @ (w * ta)}
        # diagrams 2, 3 with the couplings at the diquark's (complex) virtuality kd^2 if a table is given
        kd2 = dot(kd, kd)
        if callable(self.c):
            caa = self.c('aa', kd2)
            c1sa, c2sa = self.c('c1_sa', kd2), self.c('c2_sa', kd2)
            c1as, c2as = self.c('c1_as', kd2), self.c('c2_as', kd2)
        else:
            caa = np.full(len(kd2), self.c['aa'])
            c1sa, c1as = np.full(len(kd2), self.c['sa']), np.full(len(kd2), self.c['as'])
            c2sa = np.full(len(kd2), self.c.get('sa_long', 0.0))
            c2as = np.full(len(kd2), self.c.get('as_long', 0.0))
        # diagram 2: D1 (c_aa eps.n.kd) D1
        SPa = Sq(kq)[:, None] @ Pa                                         # (N, b, 4, 4)
        Gaa = caa[:, None, None] * np.einsum('mabv,m,nv->nab', EPS, n, kd)
        c1 = D1 @ Gaa @ D1
        taa = f2['a'] * np.einsum('wx,naxy,nab,nbyw->n', X, Ca, c1, SPa)
        # diagram 3: n_mu Gamma^sa_{mu s} = c1 n_s + c2 (n.kd) kd_s / kd^2
        nk = (kd @ n) / kd2
        vsa = c1sa[:, None] * n[None, :] + (c2sa * nk)[:, None] * kd
        vas = c1as[:, None] * n[None, :] + (c2as * nk)[:, None] * kd
        nD1 = np.einsum('ns,nsb->nb', vsa, D1)
        D1n = np.einsum('nar,nr->na', D1, vas)
        tsa = f3['sa'] * D0 * np.einsum('wx,nxy,nb,nbyw->n', X, Cs, nD1, SPa)
        SPs = Sq(kq) @ Ps
        tas = f3['as'] * D0 * np.einsum('wx,naxy,na,nyw->n', X, Ca, D1n, SPs)
        wd = self._w(kd, P)
        out['D_aa'] = wd @ (w * taa)
        out['D_sa'] = wd @ (w * tsa)
        out['D_as'] = wd @ (w * tas)
        return out

    def two_loop(self, P):
        """Flavour-stripped weighted Tr[X n.J_5] of diagram 4: res[XY]['ex4'] -> (nlam, m+1)."""
        ex, V, n = self.ex, self.V, self.n
        out, inn, kq_all, kqq_all, lq, lqq = ex._outer_inner(P, P)
        T, TV, Xp = self.structures(P)
        gam = {'s': GAM_S[None], 'a': GAM_A}
        bar = {'s': BAR_S[None], 'a': BAR_A}
        blocks = [X + Y for X in 'sa' for Y in 'sa']
        Nl = len(ex.wl)
        Tt = n[None, :, None, None] * Xp[None, None]                    # (b=1, mu, 4, 4)
        Bp = {}
        for XY in blocks:
            X, Y = XY
            B = np.einsum('mab,nrbc->nmrac', bar[X], inn[Y])
            Bp[XY] = B.transpose(0, 3, 1, 2, 4).reshape(Nl * 4, -1)
        acc = {XY: {'ex4': 0} for XY in blocks}
        nP = n @ P
        for ik in range(len(ex.wk)):
            kq, kqq = kq_all[ik], kqq_all[ik]
            q = lqq - kq
            qp = kqq - lq
            rel_in, rel_out = (kq - q) / 2, (lq - qp) / 2
            chiT = np.swapaxes(axial_quark(q), -1, -2)                    # (Nl, mu, 4, 4); Q = 0: qp = q
            wts = self._wx((q @ n) / nP, dot(q, q))
            for XY in blocks:
                X, Y = XY
                A = np.einsum('bmxy,pyz->bmpxz', Tt, out[X][ik])
                Ap = np.einsum('bmpxy,ryz->prxbmz', A, gam[Y])
                Ap = Ap.reshape(-1, Ap.shape[3] * Ap.shape[4] * 4)
                W = (Bp[XY] @ Ap).reshape(Nl, 4, 1, 4, 4)
                pref = -4 * ex.g[X] * ex.g[Y] * ex.wk[ik]
                FYin, FXout = ex.F(Y, dot(rel_in, rel_in)), ex.F(X, dot(rel_out, rel_out))
                mid = (FYin * FXout)[:, None, None, None] * chiT
                t = np.einsum('nabmc,nmca->n', W, mid)
                acc[XY]['ex4'] = acc[XY]['ex4'] + pref * (wts @ (ex.wl * t))
        return acc

    def helicity_moments(self, P, z=None, two=None, phase=None):
        """<x^m>_{Delta u}, <x^m>_{Delta d} (proton) with diagram breakdown.
        z: {'a': <z^m> of the 1+ quark-in-diquark DF, 'sa': transition DF} (default 2^-m)."""
        if z is None:
            zz = 0.5 ** self.ms
            z = {'a': zz, 'sa': zz}
        NB = self.norm(P)[0]
        T, TV, X = self.structures(P)
        hB = NB / np.trace(TV)                     # coefficient of Lambda_+ gamma.n Lambda_+ in J^B
        nT = np.trace(X @ T)
        if two is None and self.ex is not None:
            two = self.two_loop(P)
        res = {}
        for fl, tau in (('u', np.diag([1., 0.])), ('d', np.diag([0., 1.]))):
            o = self.one_loop(P, tau)
            d = {'1s': o['q_s'], '1a': o['q_a'], '2': o['D_aa'] * z['a'],
                 '3': (o['D_sa'] + o['D_as']) * z['sa'], '4': 0}
            if two is not None:
                cx = exchange_flavour(tau, np.array([1., 0.]))
                for XY, pcs in two.items():
                    d['4'] = d['4'] + cx[XY]['ex4'] * pcs['ex4']
            r = {k: v / nT / hB for k, v in d.items()}
            r['total'] = sum(r[k] for k in ('1s', '1a', '2', '3', '4'))
            r['diquark_y'] = {k: o[k] / nT / hB for k in ('D_aa', 'D_sa', 'D_as')}
            res[fl] = r
        if phase is None:                          # fix the overall phase by g_A = Delta u - Delta d > 0
            gA = res['u']['total'] - res['d']['total']
            g0 = gA[0, 0] if np.ndim(gA) == 2 else gA[0]
            phase = abs(g0) / g0
        for fl in res:
            for k, v in res[fl].items():
                res[fl][k] = {c: w * phase for c, w in v.items()} if isinstance(v, dict) else v * phase
            if len(self.lams) == 1:
                res[fl] = {k: (v[0] if not isinstance(v, dict) else {c: w[0] for c, w in v.items()})
                           for k, v in res[fl].items()}
        res['phase'] = phase
        return res
