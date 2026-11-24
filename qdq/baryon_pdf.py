"""Hadron-scale valence DFs (Mellin moments) of octet baryons: proton (check), Lambda, Sigma^0, from the
flavour-resolved quark+diquark Faddeev amplitude of qdq/baryon_faddeev.py.  Generalises qdq/pdf.py
(NucleonPDFMoments), whose construction, conventions and numerics it keeps:

    <x^m>_q = Tr[n.J^(m)_q] / Tr[n.J^B],   q = u, d, s,

n.J^(m)_q = forward (Q = 0) number current of flavour q, every insertion weighted by the light-front
fraction (n.p_struck / n.P)^m [times the regulator (1 + lam p^2)^-m], n light-like, and the canonical
normalisation Tr[n.J^B] = (n.P/M) (dlambda/dM) n0 Tr[Lambda_+]  with  int Psibar G0 Psi = n0 Lambda_+.

What changes with respect to the nucleon:
  * channel groups g (diquark type J x diquark class ll/ls/ss x bystander class l/s; BARYONS in
    baryon_faddeev) replace X = s, a; every group has its own diquark mass, width and coupling, and its
    own bystander propagator (light or strange); the exchanged quark is light or strange (class e);
  * forward vertices: quark i chi_mu(p, p) = -dS_f/dp_mu with the flavour's propagator (same numerics as
    current.Vertices.chi_quark at l1 = l2), diquark -dDelta_g/dK_mu (current.Vertices with the group's
    mass); the 0+ <-> 1+ transition (diagram 3) vanishes at Q = 0, as for the nucleon;
  * flavour x charge factors from the group flavour tensors F^g_{ab,c} (diquark quarks a, b; bystander c)
    with a diagonal flavour charge t = t^q:
       diagram 1 (bystander):      sum F^g_{ab,c} t_c F^g'_{ab,c}
       diagram 2 (diquark):        sum F^g_{ab,c} (t_a + t_b) F^g'_{ab,c}       [split by the struck quark]
       diagrams 4-6, exchanged quark j (pieces ex4, ex5, ex6):  sum_{i,k} sum_{j in e} F^g_{kj,i} t_j F^g'_{ij,k}
                     outgoing bystander i (by5, seagull on Gamma^{g'}):       ... t_i ...
                     incoming bystander k (by6, seagull on Gammabar^{g}):     ... t_k ...
    (with t = 1 these are the kernel's c^{(e)}_{g g'}; for the proton they equal current.flavour_factors
    and exchange.exchange_flavour);
  * diagram 2 is convolved with the quark-in-diquark DF of the STRUCK quark: for an ls diquark the light
    and the strange quark have different distributions, q_l(z) ~ phi(z)^2 and q_s(z) ~ phi(1-z)^2 with
    phi the asymmetric diquark DA (z = light-quark fraction; qdq/diquark_df.da_moments_flavour).
    z_moments is keyed by (J, diquark class, struck class), e.g. ('s', 'ls', 's').
Checks: m = 0 gives the valence numbers (p: u, d = 2, 1;  Lambda, Sigma^0: u = d = s = 1) through the
Ward identity of the flavour-resolved kernel; sum_q <x>_q = 1 (momentum sum, with <z>_l + <z>_s = 1);
the proton from BaryonFaddeev reproduces NucleonPDFMoments (6e-12); in the SU(3) limit the octet relations
(Sigma0: u = u_p/2, s = d_p; Lambda: u = (u_p + 4 d_p)/6, s = (2 u_p - d_p)/3) hold diagram by diagram
(1e-12; scripts/18_hyperon_pdf.py --su3).
"""
import numpy as np
from .dirac import G, I4, C, Cdag, slash
from .quark import calF, sigmas
from .quadrature import radial, chebyshev_u
from .faddeev import basis, dot, GAM_S, GAM_A, BAR_S, BAR_A
from .baryon_faddeev import quark_S, U, D, S
from .current import lam_plus, Vertices

NV = np.array([0, 0, 1, 1j])                       # light-like: n.n = 0
FLAV = {'u': U, 'd': D, 's': S}
CLASS = {'u': 'l', 'd': 'l', 's': 's'}
PIECES = ['ex4', 'by5', 'ex5', 'by6', 'ex6']


def _der(f, x, h=1e-6):
    """Central derivative, as current.divdiff at x1 = x2."""
    x = np.asarray(x, complex)
    return (f(x + h) - f(x - h)) / (2 * h)


def quark_vertex_fwd(p, par):
    """Forward number-current vertex i chi_mu(p, p) = -dS_f/dp_mu: (N,4) -> (N, mu, 4, 4)."""
    p = np.asarray(p, complex)
    a = dot(p, p)
    sV = lambda x: sigmas(x, **par)[0]
    sS = lambda x: sigmas(x, **par)[1]
    DV, DS = _der(sV, a), _der(sS, a)
    chi = G * sV(a)[..., None, None, None]
    bracket = slash(p) * DV[..., None, None] + 1j * DS[..., None, None] * I4
    chi = chi + 2 * p[..., :, None, None] * bracket[..., None, :, :]
    return 1j * chi


def grid3(nk, nz, ny, scale):
    """Outer/one-loop grid of NucleonCurrent / ExchangeCurrent (azimuth fixed, factor 2 pi)."""
    r, wr = radial(nk, scale); z, wz = chebyshev_u(nz); y, wy = np.polynomial.legendre.leggauss(ny)
    R, Z, Y = np.meshgrid(r, z, y, indexing='ij')
    sz, sy = np.sqrt(1 - Z**2), np.sqrt(1 - Y**2)
    k = np.stack([R * sz * sy, 0 * R, R * sz * Y, R * Z], -1).reshape(-1, 4)
    w = (wr[:, None, None] * r[:, None, None]**3 * wz[None, :, None] * wy[None, None, :]
         * 2 * np.pi / (2 * np.pi)**4).reshape(-1)
    return k.astype(complex), w


def grid4(nl, nz, ny, nph, scale):
    """Inner grid of ExchangeCurrent with explicit relative azimuth."""
    r, wr = radial(nl, scale); z, wz = chebyshev_u(nz); y, wy = np.polynomial.legendre.leggauss(ny)
    ph = 2 * np.pi * (np.arange(nph) + 0.5) / nph
    R, Z, Y, PH = np.meshgrid(r, z, y, ph, indexing='ij')
    sz, sy = np.sqrt(1 - Z**2), np.sqrt(1 - Y**2)
    l = np.stack([R * sz * sy * np.cos(PH), R * sz * sy * np.sin(PH), R * sz * Y, R * Z], -1).reshape(-1, 4)
    w = (wr[:, None, None, None] * r[:, None, None, None]**3 * wz[None, :, None, None]
         * wy[None, None, :, None] * np.full(nph, 2 * np.pi / nph) / (2 * np.pi)**4).reshape(-1)
    return l.astype(complex), w


# ------------------------------------------------------------------ flavour ----
def charge_factors(groups, q):
    """Flavour x charge factors for the number current of flavour q ('u', 'd', 's').
    Returns f1[g, g'], f2[g, g'] (arrays) and fx[(g, g', e)][piece] (dict)."""
    t = np.zeros(3); t[FLAV[q]] = 1.0
    n = len(groups)
    f1, f2 = np.zeros((n, n)), np.zeros((n, n))
    fx = {}
    for ig, g in enumerate(groups):
        for jg, gp in enumerate(groups):
            Fg, Fp = g['F'], gp['F']
            f1[ig, jg] = np.einsum('abc,c,abc->', Fg, t, Fp)
            f2[ig, jg] = np.einsum('abc,a,abc->', Fg, t, Fp) + np.einsum('abc,b,abc->', Fg, t, Fp)
            for e, js in (('l', (U, D)), ('s', (S,))):
                mask = np.zeros(3); mask[list(js)] = 1.0
                # sum_{i,k} sum_{j in e} F^g_{kj,i} [charge] F^g'_{ij,k}
                ex = np.einsum('kji,j,j,ijk->', Fg, mask, t, Fp)
                b5 = np.einsum('kji,j,i,ijk->', Fg, mask, t, Fp)
                b6 = np.einsum('kji,j,k,ijk->', Fg, mask, t, Fp)
                fx[ig, jg, e] = {'ex4': ex, 'by5': b5, 'ex5': ex, 'by6': b6, 'ex6': ex}
    return f1, f2, fx


def struck_split(group, q):
    """Diagram-2 flavour factor of group g for flavour q (struck quark of flavour q in the diquark)."""
    t = np.zeros(3); t[FLAV[q]] = 1.0
    F = group['F']
    return np.einsum('abc,a,abc->', F, t, F) + np.einsum('abc,b,abc->', F, t, F)


# --------------------------------------------------------------- amplitudes ----
class BaryonAmplitudes:
    """Dirac amplitudes Psi_g(l; P) (Lambda_+ included) of a solved BaryonFaddeev, per group:
    (N, 1, 4, 4) for a 0+ group and (N, 4, 4, 4) [Lorentz index first] for a 1+ group."""

    def __init__(self, fad):
        assert fad.M is not None, "solve the Faddeev equation first"
        self.fad, self.M, self.eta = fad, fad.M, fad.eta
        self.groups = fad.groups
        self.V = [Vertices((fad.mass[g['J'], g['dq']],) * 2, fad.M) for g in self.groups]

    def amplitude(self, l, P):
        f = self.fad.scalar_functions(l, P)
        lhat = l / np.sqrt(dot(l, l)).real[..., None]
        Bs, Ba = basis(lhat, P / self.M)
        L = lam_plus(P, self.M)
        out = []
        for ig, g in enumerate(self.groups):
            fg = f[:, self.fad.off[ig]:self.fad.off[ig + 1]]
            if g['J'] == 's':
                out.append((np.einsum('nc,ncab->nab', fg, Bs) @ L)[:, None])
            else:
                out.append(np.einsum('nc,ncvab->nvab', fg, Ba) @ L)
        return out

    def conj_amplitude(self, p, P):
        return [Cdag @ np.swapaxes(A, -1, -2) @ C for A in self.amplitude(-p, -P)]

    def Sby(self, ig, p):
        return quark_S(p, self.fad.q[self.groups[ig]['by']])

    def Delta(self, ig, K):
        """Diquark propagator of group ig: (N, 1, 1) or (N, 4, 4)."""
        V = self.V[ig]
        if self.groups[ig]['J'] == 's':
            return V.D0(dot(K, K))[:, None, None]
        return V.D1mat(K)

    def dDelta_n(self, ig, K, n):
        """n.(-dDelta/dK) at K (forward diquark number vertex): (N, 1, 1) or (N, 4, 4)."""
        V = self.V[ig]
        if self.groups[ig]['J'] == 's':
            return (V.chi_s(K, K) @ n)[:, None, None]
        return np.einsum('m,nmab->nab', n, V.chi_a(K, K))


# -------------------------------------------------------------------- moments ----
class BaryonPDFMoments:
    def __init__(self, fad, one=(48, 32, 24), outer=(12, 6, 6), inner=(12, 6, 6, 6), scale=0.6,
                 mmax=6, n=NV, kmax=15.0, lams=(0.0,)):
        self.fad = fad
        self.amp = BaryonAmplitudes(fad)
        self.groups = fad.groups
        self.M, self.eta = fad.M, fad.eta
        self.mmax, self.n = mmax, np.asarray(n, complex)
        self.ms = np.arange(mmax + 1)
        self.lams = np.asarray(lams, float)
        kmax = np.inf if kmax is None else kmax
        cut = lambda v, w: (v[np.linalg.norm(v.real, axis=1) < kmax], w[np.linalg.norm(v.real, axis=1) < kmax])
        self.k1, self.w1 = cut(*grid3(*one, scale))
        self.kx, self.wkx = cut(*grid3(*outer, scale)) if outer else (None, None)
        self.lx, self.wlx = cut(*grid4(*inner, scale)) if inner else (None, None)

    def _wx(self, x, p2):
        r = x[None, :] / (1 + self.lams[:, None] * p2[None, :])
        return r[:, None, :] ** self.ms[None, :, None]

    def _w(self, p, P):
        return self._wx((p @ self.n) / (self.n @ P), dot(p, p))

    # ------------------------------------------------------- normalisation ----
    def psibar_G0_psi(self, P):
        A, eta = self.amp, self.eta
        k, w = self.k1, self.w1
        Ps, Cs = A.amplitude(k, P), A.conj_amplitude(k, P)
        kq, kd = eta * P + k, (1 - eta) * P - k
        tot = 0
        per = []
        for ig in range(len(self.groups)):
            t = np.einsum('naxy,nyz,nab,nbzx->n', Cs[ig], A.Sby(ig, kq), A.Delta(ig, kd), Ps[ig])
            per.append((t @ w) / 2)
            tot = tot + per[-1]
        return tot, per

    def norm(self, P, dM=1e-3):
        lp = self.fad.eigen(self.M + dM)[0]
        lm = self.fad.eigen(self.M - dM)[0]
        dlam = (lp - lm) / (2 * dM)
        n0, per = self.psibar_G0_psi(P)
        return (self.n @ P) / self.M * dlam * n0 * 2, dlam, n0, per

    # ------------------------------------------------------------ one loop ----
    def one_loop(self, P):
        """Flavour-stripped weighted traces per group: q[g] (probe on the bystander, weight k_q) and
        D[g] (probe on the diquark, weight k_d) -> (nlam, mmax+1) arrays."""
        A, eta, n = self.amp, self.eta, self.n
        k, w = self.k1, self.w1
        Ps, Cs = A.amplitude(k, P), A.conj_amplitude(k, P)
        kq, kd = eta * P + k, (1 - eta) * P - k
        wq, wd = self._w(kq, P), self._w(kd, P)
        out = {'q': [], 'D': []}
        for ig, g in enumerate(self.groups):
            vq = np.einsum('m,nmab->nab', n, quark_vertex_fwd(kq, self.fad.q[g['by']]))
            Dk = A.Delta(ig, kd)
            t1 = np.einsum('naxy,nyz,nab,nbzx->n', Cs[ig], vq, Dk, Ps[ig])
            t2 = np.einsum('naxy,nyz,nab,nbzx->n', Cs[ig], A.Sby(ig, kq), A.dDelta_n(ig, kd, n), Ps[ig])
            out['q'].append(wq @ (w * t1))
            out['D'].append(wd @ (w * t2))
        return out

    # ------------------------------------------------------------ two loop ----
    def blocks(self):
        """(g, g', e) combinations with a non-zero flavour factor for some flavour and piece."""
        bl = set()
        for q in 'uds':
            _, _, fx = charge_factors(self.groups, q)
            for key, d in fx.items():
                if any(abs(v) > 1e-14 for v in d.values()):
                    bl.add(key)
        return sorted(bl)

    def two_loop(self, P, verbose=False):
        """Flavour-stripped weighted traces of diagrams 4-6: acc[(g, g', e)][piece] -> (nlam, mmax+1)."""
        import time
        A, eta, n, fad = self.amp, self.eta, self.n, self.fad
        k, wk, l, wl = self.kx, self.wkx, self.lx, self.wlx
        Cg, Pg = A.conj_amplitude(k, P), A.amplitude(l, P)
        kq_all, kqq_all = eta * P + k, (1 - eta) * P - k
        lq, lqq = eta * P + l, (1 - eta) * P - l
        ng = len(self.groups)
        out = [np.einsum('nam,naxy,nyz->nmxz', A.Delta(ig, kqq_all), Cg[ig], A.Sby(ig, kq_all)) for ig in range(ng)]
        inn = [np.einsum('nxy,nrv,nvyz->nrxz', A.Sby(ig, lq), A.Delta(ig, lqq), Pg[ig]) for ig in range(ng)]
        gam = {'s': GAM_S[None], 'a': GAM_A}
        bar = {'s': BAR_S[None], 'a': BAR_A}
        J = [g['J'] for g in self.groups]
        coup = [fad.g[g['J'], g['dq']] for g in self.groups]
        w2 = [fad._w2(g) for g in self.groups]
        blocks = self.blocks()
        Nl = len(wl)
        Bp = {}
        for ig, jg, e in blocks:
            if (ig, jg) not in Bp:
                B = np.einsum('mab,nrbc->nmrac', bar[J[ig]], inn[jg])
                Bp[ig, jg] = B.transpose(0, 3, 1, 2, 4).reshape(Nl * 4, -1)
        acc = {b: {pc: 0 for pc in PIECES} for b in blocks}
        es = sorted({e for (_, _, e) in blocks})
        nP = n @ P
        t0 = time.time()
        for ik in range(len(wk)):
            kq, kqq = kq_all[ik], kqq_all[ik]
            q = lqq - kq
            qp = kqq - lq
            rel_in, rel_out = (kq - q) / 2, (lq - qp) / 2
            l5, l6 = (kq - qp) / 2, (lq - q) / 2
            n5, n6 = l5 @ n, l6 @ n
            kqf = np.broadcast_to(kq, q.shape)
            mom = {'ex4': q, 'by5': kqf, 'ex5': qp, 'by6': lq, 'ex6': q}
            wts = {pc: self._wx((pp @ n) / nP, dot(pp, pp)) for pc, pp in mom.items()}
            Se = {}
            for e in es:
                par = fad.q[e]
                chin = np.einsum('m,nmab->nab', n, quark_vertex_fwd(q, par))
                Se[e] = (np.swapaxes(chin, -1, -2), np.swapaxes(quark_S(q, par), -1, -2),
                         np.swapaxes(quark_S(qp, par), -1, -2))
            ri2, ro2, l52, l62 = dot(rel_in, rel_in), dot(rel_out, rel_out), dot(l5, l5), dot(l6, l6)
            for ig, jg, e in blocks:
                chiT, SqT, SqpT = Se[e]
                Ap = np.einsum('pxy,ryz->prxz', out[ig][ik], gam[J[jg]]).reshape(-1, 4)
                W = (Bp[ig, jg] @ Ap).reshape(Nl, 4, 4)                      # [n, a, c]
                pref = -4 * coup[ig] * coup[jg] * wk[ik]
                FY = lambda x: calF(x / w2[jg])
                FX = lambda x: calF(x / w2[ig])
                FYin, FXout = FY(ri2), FX(ro2)
                d5, d6 = _der(FY, l52), _der(FX, l62)
                mids = {
                    'ex4': (FYin * FXout)[:, None, None] * chiT,
                    'by5': (-n5 * d5 * FXout)[:, None, None] * SqpT,
                    'ex5': (n5 * d5 * FXout)[:, None, None] * SqpT,
                    'by6': (-n6 * d6 * FYin)[:, None, None] * SqT,
                    'ex6': (n6 * d6 * FYin)[:, None, None] * SqT,
                }
                for pc, mid in mids.items():
                    t = np.einsum('nac,nca->n', W, mid)
                    acc[ig, jg, e][pc] = acc[ig, jg, e][pc] + pref * (wts[pc] @ (wl * t))
            if verbose and ik % 50 == 0:
                print(f"      two-loop outer point {ik}/{len(wk)}  ({time.time() - t0:.0f}s)", flush=True)
        return acc

    # ------------------------------------------------------------ assemble ----
    def moments(self, P, z_moments=None, one=None, two=None, nb=None):
        """<x^m>_q for q = u, d, s with the diagram breakdown.
        z_moments: {(J, dq, struck class): (mmax+1,)}; default 2^-m (only m <= 1 meaningful, and for
        ls diquarks only m = 0)."""
        if z_moments is None:
            z_moments = {}
        zdef = 0.5 ** self.ms
        NB, dlam, n0, per = self.norm(P) if nb is None else nb
        one = self.one_loop(P) if one is None else one
        two = self.two_loop(P) if (two is None and self.kx is not None) else (two or {})
        res = {}
        for q in 'uds':
            f1, f2, fx = charge_factors(self.groups, q)
            off = f1 - np.diag(np.diag(f1)), f2 - np.diag(np.diag(f2))
            assert max(np.abs(off[0]).max(), np.abs(off[1]).max()) < 1e-12, "one-loop factors not diagonal"
            d = {'1': 0, '2': 0, '4': 0, '5': 0, '6': 0}
            ydq = {}
            for ig, g in enumerate(self.groups):
                d['1'] = d['1'] + f1[ig, ig] * one['q'][ig]
                z = z_moments.get((g['J'], g['dq'], CLASS[q]), zdef)
                d['2'] = d['2'] + f2[ig, ig] * one['D'][ig] * z
                ydq[g['name']] = f2[ig, ig] * one['D'][ig] / NB
            for key, pcs in two.items():
                for pc, val in pcs.items():
                    d[pc[-1]] = d[pc[-1]] + fx[key][pc] * val
            res[q] = {kk: v / NB for kk, v in d.items()}
            res[q]['total'] = sum(res[q][kk] for kk in '12456')
            res[q]['diquark_y'] = ydq
            if len(self.lams) == 1:
                res[q] = {kk: (v[0] if not isinstance(v, dict) else {c: w[0] for c, w in v.items()})
                          for kk, v in res[q].items()}
        res['norm'] = dict(NB=NB, dlam=dlam, n0=n0, n0_groups=per)
        return res
