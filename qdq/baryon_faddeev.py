"""Quark + diquark Faddeev equation for octet baryons with u, d, s quarks (isospin symmetric):
proton (check), Lambda, Sigma^0.  Generalises qdq/faddeev.py (NucleonFaddeev), whose structure and
conventions it keeps; with m_s = m_u and equal diquark masses it must reproduce the nucleon.

Kernel (same as qdq/faddeev.py, now with flavour-dependent propagators):
  Phi_g(k) = -4 sum_{g'} sum_{e} c^{(e)}_{g g'} g_g g_{g'} int d^4l/(2pi)^4
             Gamma^{g'}(k_q - l_qq/2) S_e^T(l_qq - k_q) Gammabar^g(l_q - k_qq/2) S_{b(g')}(l_q) Delta^{g'}(l_qq) Phi_{g'}(l)
  g, g'   channel groups: diquark type J (0+ / 1+) x diquark flavour class (ll, ls, ss) x bystander class
          (l, s); the isospin partners inside a group share one Dirac amplitude (2 or 6 scalar functions);
  e       flavour class (l, s) of the exchanged quark;  b(g') bystander class of the incoming group.
Flavour factors (generalising c_XY of qdq/faddeev.py, which they reproduce for the proton): with the
group flavour tensor F^g_{ab,c} = sum coeff T^D_{ab} delta_{c f} (diquark quarks a, b; bystander c),
    c^{(e)}_{g g'} = sum_{i,k} sum_{j in e} F^g_{kj,i} F^{g'}_{ij,k}
(outgoing bystander i; incoming diquark (i, j); exchanged quark j; incoming bystander k).
Diquark flavour matrices: [ab] = (ab - ba)/sqrt2, {ab} = (ab + ba)/sqrt2, {aa} = aa (unit norm).
Momentum partition inside every diquark: equal (relative momentum (p_a - p_b)/2), as in qdq/faddeev.py;
quark-diquark partition eta = 1/3 for every channel (exact results do not depend on it).

Inputs: light quark = 2611.19819 App. A.1; strange quark = Chen, Krein, Roberts, Schmidt,
arXiv:1901.04305 App. A (same lambda = 0.566 GeV; m_bar = 0.210, b = 0.105, 3.18, 0.858, 0.185).
Diquark masses: defaults from the equal-spacing rule of 1901.04305 (+0.15 GeV per s quark) applied to
the 2611.19819 values 0.79 / 0.89 GeV.  Couplings from the canonical normalisation (diquark_coupling),
or given explicitly.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from .dirac import G, G5, C, Cdag, I4, slash
from .quark import sigmas, calF
from .quadrature import radial, chebyshev_u, loop4d
from .faddeev import basis, cheb_u, dot, GAM_S, GAM_A, BAR_S, BAR_A, NucleonFaddeev

QUARKS = {'l': dict(mbar=0.00897, b0=0.131, b1=2.90, b2=0.603, b3=0.185),
          's': dict(mbar=0.210, b0=0.105, b1=3.18, b2=0.858, b3=0.185)}   # 1901.04305 Eq. (A.6)
MASSES = {('s', 'll'): 0.79, ('a', 'll'): 0.89,
          ('s', 'ls'): 0.94, ('a', 'ls'): 1.04, ('a', 'ss'): 1.19}      # GeV; ESR +0.15 per s
U, D, S = 0, 1, 2


def quark_S(p, par):
    """S_f(p) = -i gamma.p sigma_V + sigma_S for momenta (...,4)."""
    p = np.asarray(p, dtype=complex)
    sV, sS = sigmas(dot(p, p), **par)
    return -1j * slash(p) * sV[..., None, None] + sS[..., None, None] * I4


# ------------------------------------------------------------------ flavour ----
def anti(a, b):
    T = np.zeros((3, 3)); T[a, b] = 1 / np.sqrt(2); T[b, a] = -1 / np.sqrt(2); return T

def sym(a, b):
    T = np.zeros((3, 3))
    if a == b:
        T[a, a] = 1.0
    else:
        T[a, b] = T[b, a] = 1 / np.sqrt(2)
    return T

def _cls(f):
    return 's' if f == S else 'l'

def make_group(name, J, comps):
    """comps: list of (diquark flavour matrix T, bystander flavour, coefficient)."""
    F = np.zeros((3, 3, 3))
    dq, by = set(), set()
    for T, f, c in comps:
        e = np.zeros(3); e[f] = 1.0
        F += c * T[:, :, None] * e[None, None, :]
        a, b = np.argwhere(np.abs(T) > 0)[0]
        dq.add(''.join(sorted(_cls(a) + _cls(b))))
        by.add(_cls(f))
    assert len(dq) == 1 and len(by) == 1, f"group {name}: mixed masses"
    assert abs(np.sum(F**2) - 1) < 1e-12, f"group {name}: not unit-normalised"
    return dict(name=name, J=J, dq=dq.pop(), by=by.pop(), F=F, nb=2 if J == 's' else 6)

BARYONS = {
    'p': [('[ud]u', 's', [(anti(U, D), U, 1.0)]),
          ('{uu}d,{ud}u', 'a', [(sym(U, U), D, np.sqrt(2 / 3)), (sym(U, D), U, -np.sqrt(1 / 3))])],
    'Lambda': [('[ud]s', 's', [(anti(U, D), S, 1.0)]),
               ('[us]d-[ds]u', 's', [(anti(U, S), D, 1 / np.sqrt(2)), (anti(D, S), U, -1 / np.sqrt(2))]),
               ('{us}d-{ds}u', 'a', [(sym(U, S), D, 1 / np.sqrt(2)), (sym(D, S), U, -1 / np.sqrt(2))])],
    'Sigma0': [('[us]d+[ds]u', 's', [(anti(U, S), D, 1 / np.sqrt(2)), (anti(D, S), U, 1 / np.sqrt(2))]),
               ('{ud}s', 'a', [(sym(U, D), S, 1.0)]),
               ('{us}d+{ds}u', 'a', [(sym(U, S), D, 1 / np.sqrt(2)), (sym(D, S), U, 1 / np.sqrt(2))])],
}

def flavour_coeffs(groups):
    """c[(g, g', e)] for groups g (out), g' (in) and exchanged-quark class e."""
    out = {}
    for ig, g in enumerate(groups):
        for jg, gp in enumerate(groups):
            for e, js in (('l', (U, D)), ('s', (S,))):
                v = sum(g['F'][k, j, i] * gp['F'][i, j, k] for i in range(3) for k in range(3) for j in js)
                if abs(v) > 1e-14:
                    out[ig, jg, e] = v
    return out


# --------------------------------------------------------- diquark couplings ----
def diquark_coupling(J, m, qa, qb, nq=160, nz=24, h=1e-3):
    """Canonical normalisation (qdq/diquark.py coupling) for a diquark of type J ('s'/'a'), mass m,
    made of quarks with propagator parameters qa (first index) and qb; equal momentum partition.
    Colour-flavour trace 2 (unit-norm flavour matrix x colour antitriplet)."""
    qv, W = loop4d(nq, nz, scale=1.0)
    w2 = 0.5 * m**2
    F2 = calF(dot(qv, qv) / w2) ** 2
    K = np.array([0, 0, 0, 1j * m])
    def Pi(Q):
        Sp = quark_S(qv + Q / 2, qa)
        SmT = np.swapaxes(quark_S(-qv + Q / 2, qb), -1, -2)
        if J == 's':
            v = np.einsum('ab,...bc,cd,...da->...', BAR_S, Sp, GAM_S, SmT)
            return np.sum(W * F2 * v)
        T = np.eye(4) - np.outer(K, K) / (-m**2)
        tot = 0
        for a in range(4):
            for b in range(4):
                if abs(T[a, b]) > 0:
                    v = np.einsum('ab,...bc,cd,...da->...', BAR_A[a], Sp, GAM_A[b], SmT)
                    tot += T[a, b] * np.sum(W * F2 * v)
        return tot
    lhs = 2 * (-m**2) * (1 if J == 's' else 3)
    dPi = (Pi((1 + h) * K) - Pi((1 - h) * K)) / (2 * h) * 2.0
    g2 = lhs / dPi
    return np.sqrt(complex(g2.real)), g2


# ------------------------------------------------------------------- solver ----
class BaryonFaddeev:
    """Octet-baryon Faddeev equation.  Unknowns f[c, r_i, n]: c runs over the scalar functions of all
    groups (2 per 0+ group, 6 per 1+ group, in group order)."""

    def __init__(self, baryon='p', quarks=None, masses=None, couplings=None, nr=16, nz=8, ny=8,
                 ncheb=6, scale=0.6, eta=1 / 3):
        self.baryon = baryon
        self.groups = [make_group(*g) for g in BARYONS[baryon]]
        self.q = dict(QUARKS, **(quarks or {}))
        self.mass = {**MASSES, **(masses or {})}
        self.eta = eta
        self.c = flavour_coeffs(self.groups)
        # couplings per (J, dq): given, or from the canonical normalisation
        self.g = {}
        for gr in self.groups:
            key = (gr['J'], gr['dq'])
            if couplings and key in couplings:
                self.g[key] = couplings[key]
            elif key not in self.g:
                qa = self.q['l'] if gr['dq'][0] == 'l' else self.q['s']
                qb = self.q['l'] if gr['dq'][1] == 'l' else self.q['s']
                self.g[key] = diquark_coupling(gr['J'], self.mass[key], qa, qb)[0]
        self.off = np.cumsum([0] + [g['nb'] for g in self.groups])
        self.nch = int(self.off[-1])
        # grids (as NucleonFaddeev)
        self.r, self.wr = radial(nr, scale)
        self.z, self.wz = chebyshev_u(nz)
        self.y, self.wy = np.polynomial.legendre.leggauss(ny)
        self.nr, self.nz, self.ny, self.ncheb = nr, nz, ny, ncheb
        if ncheb is None:
            self.zext, wext = self.z, self.wz
            self.Pout, self.Ein = np.eye(nz), np.eye(nz)
        else:
            assert ncheb <= nz
            self.zext, wext = chebyshev_u(ncheb)
            self.Pout = 2 / np.pi * wext[None, :] * cheb_u(ncheb, self.zext).real
            self.Ein = cheb_u(ncheb, self.z).real.T
        self.nzext = len(self.zext)
        R, Z = np.meshgrid(self.r, self.zext, indexing='ij')
        self.kvec = np.stack([0 * R, 0 * R, R * np.sqrt(1 - Z**2), R * Z], -1)
        R3, Z3, Y3 = np.meshgrid(self.r, self.z, self.y, indexing='ij')
        sz, sy = np.sqrt(1 - Z3**2), np.sqrt(1 - Y3**2)
        self.lvec = np.stack([0 * R3, R3 * sz * sy, R3 * sz * Y3, R3 * Z3], -1).reshape(-1, 4)
        self.wl = (self.wr[:, None, None] * self.r[:, None, None]**3 * self.wz[None, :, None]
                   * self.wy[None, None, :] * 2 * np.pi / (2 * np.pi)**4).reshape(-1)
        self.Phat = np.array([0, 0, 0, 1j])
        self.Lp = 0.5 * (I4 + G[3])
        lhat = self.lvec / np.linalg.norm(self.lvec, axis=-1, keepdims=True)
        Bs_l, Ba_l = basis(lhat, self.Phat)
        self.Bs_l, self.Ba_l = Bs_l @ self.Lp, Ba_l @ self.Lp
        khat = self.kvec / np.linalg.norm(self.kvec, axis=-1, keepdims=True)
        Bs_k, Ba_k = basis(khat, self.Phat)
        self.Ds, self.Da = NucleonFaddeev._duals(Bs_k @ self.Lp, Ba_k @ self.Lp)
        self.M = None

    def _w2(self, gr):
        return 0.5 * self.mass[gr['J'], gr['dq']]**2

    def kernel_matrix(self, M):
        P = np.array([0, 0, 0, 1j * M])
        l = self.lvec
        Ni = l.shape[0]
        lq, lqq = l + self.eta * P, -l + (1 - self.eta) * P
        lqq2 = dot(lqq, lqq)
        Sby = {b: quark_S(lq, self.q[b]) for b in {g['by'] for g in self.groups}}
        bar = {'s': BAR_S[None], 'a': BAR_A}
        gam = {'s': GAM_S[None], 'a': GAM_A}
        # incoming side Delta^{g'} B_n per group: (Ni, nb, rho, 4, 4)
        inp = []
        for gp in self.groups:
            m = self.mass[gp['J'], gp['dq']]
            Pi = calF(lqq2 / self._w2(gp)) / m**2
            if gp['J'] == 's':
                inp.append((Pi[:, None, None, None] * self.Bs_l)[:, :, None])
            else:
                Dlt = (np.eye(4) + lqq[:, :, None] * lqq[:, None, :] / m**2) * Pi[:, None, None]
                inp.append(np.einsum('irv,invab->inrab', Dlt, self.Ba_l))
        # inner-only pieces Rin[g, g'] = Gammabar^g S_{b(g')}(l_q) inp[g']
        Rin = {}
        pairs = sorted({(ig, jg) for (ig, jg, e) in self.c})
        for ig, jg in pairs:
            t = np.einsum('mab,ibc->imac', bar[self.groups[ig]['J']], Sby[self.groups[jg]['by']])
            R = np.matmul(t[:, None, :, None], inp[jg][:, :, None, :])
            Rin[ig, jg] = R.transpose(0, 1, 4, 2, 3, 5).reshape(Ni * R.shape[1] * 4, -1)
        nr, nza, nz, ny = self.nr, self.nzext, self.nz, self.ny
        Kfull = np.zeros((self.nch, nr, nza, self.nch, nr * nz), complex)
        for i in range(nr):
            for a in range(nza):
                k = self.kvec[i, a]
                kq, kqq = k + self.eta * P, -k + (1 - self.eta) * P
                rin, rout = kq - lqq / 2, lq - kqq / 2
                r_in, r_out = dot(rin, rin), dot(rout, rout)
                SexT = {e: np.swapaxes(quark_S(lqq - kq, self.q[e]), -1, -2) for e in {e for (_, _, e) in self.c}}
                Dd = {'s': self.Ds[i, a][:, None], 'a': self.Da[i, a]}
                for (ig, jg, e), cval in self.c.items():
                    g, gp = self.groups[ig], self.groups[jg]
                    Lout = np.einsum('kmba,rbc->kmrac', Dd[g['J']], gam[gp['J']])
                    nX, nY = g['nb'], gp['nb']
                    B1 = Lout.transpose(1, 2, 3, 0, 4).reshape(-1, nX * 4)
                    W = (Rin[ig, jg] @ B1).reshape(Ni, nY, 4, nX, 4)
                    val = np.einsum('iyz,inzky->ikn', SexT[e], W)
                    coef = (-4 * cval * self.g[g['J'], g['dq']] * self.g[gp['J'], gp['dq']]
                            * calF(r_in / self._w2(gp)) * calF(r_out / self._w2(g)) * self.wl)
                    val = (coef[:, None, None] * val).reshape(nr * nz, ny, nX, nY).sum(1)
                    Kfull[self.off[ig]:self.off[ig + 1], i, a, self.off[jg]:self.off[jg + 1]] += val.transpose(1, 2, 0)
        K = Kfull.reshape(self.nch, nr, nza, self.nch, nr, nz)
        K = np.einsum('ma,ciadjb,bn->cimdjn', self.Pout, K, self.Ein, optimize=True)
        nm = self.Pout.shape[0]
        self._nm = nm
        return K.reshape(self.nch * nr * nm, self.nch * nr * nm)

    def spectrum(self, M, n=4):
        ev = np.linalg.eigvals(self.kernel_matrix(M))
        return ev[np.argsort(-ev.real)][:n]

    def eigen(self, M):
        ev, evec = np.linalg.eig(self.kernel_matrix(M))
        j = np.argmax(ev.real)
        return ev[j], evec[:, j]

    def solve_mass(self, M0, M1, tol=1e-6, verbose=False):
        f0 = self.eigen(M0)[0].real - 1
        f1 = self.eigen(M1)[0].real - 1
        for _ in range(40):
            M2 = M1 - f1 * (M1 - M0) / (f1 - f0)
            M0, f0 = M1, f1
            lam, vec = self.eigen(M2)
            M1, f1 = M2, lam.real - 1
            if verbose:
                print(f"   M = {M1:.5f} GeV   lambda - 1 = {f1:+.2e}")
            if abs(f1) < tol:
                break
        self.M = M1
        self._store(vec)
        return M1

    def _store(self, vec):
        f = vec.reshape(self.nch, self.nr, self._nm)
        if self.ncheb is None:
            Pm = 2 / np.pi * self.wz[None, :] * cheb_u(self.nz, self.z).real
            f = np.einsum('na,cia->cin', Pm, f)
        self.f = (f / f[0, 0, 0]).transpose(0, 2, 1)                # (c, n, r)
        self.splines = CubicSpline(np.log(self.r), self.f, axis=2)

    def scalar_functions(self, l, P):
        l = np.asarray(l, dtype=complex)
        Phat = P / self.M
        L = np.sqrt(dot(l, l))
        z = -1j * dot(l, Phat) / L
        logr = np.clip(np.log(L.real), np.log(self.r[0]), np.log(self.r[-1]))
        fn = np.moveaxis(self.splines(logr), (0, 1), (-2, -1))
        Uz = np.moveaxis(cheb_u(fn.shape[-1], z), 0, -1)
        return np.einsum('...cn,...n->...c', fn, Uz)
