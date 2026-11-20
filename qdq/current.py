"""Nucleon electromagnetic current in the quark+diquark picture: one-loop diagrams 1-3.

Follows P. Cheng et al., arXiv:2507.13484 (Sect. 3), which refines Cloet et al. 0812.0416:
  Diagram 1: photon on the bystander quark, unamputated dressed quark-photon vertex chi_mu;
  Diagram 2: photon on the (elastic) scalar / axial-vector diquark, unamputated vertices;
  Diagram 3: photon-induced 0+ <-> 1+ diquark transition (|kappa_01| = 2.64; phase and relative
             sign derived from the quark loop in our conventions, see VERTEX_PARAMS).
Diagrams 4-6 (two-loop: exchange + seagulls) are milestone 5c.

Current:  J_mu = int d^4k/(2pi)^4  Psibar(p; -P_f) Lambda_mu(p, P_f; k, P_i) Psi(k; P_i),
          Psibar(p; -P) = C^dag Psi(-p; -P)^T C,     Breit frame P_{f,i} = (0,0,0,iE) +- Q/2.
All vertices obey  Q.V(l1, l2) = G(l2) - G(l1)  (G = S for quarks [with V = i chi], Delta for
diquarks), so at Q -> 0 the vertices reduce to -dG/dl.

Form factors are projected point-by-point with Lorentz-scalar traces onto
  T1_mu = L_f gamma_mu L_i,  T2_mu = L_f sigma_{mu nu} Q_nu L_i /(2M),  T3_mu = L_f Q_mu L_i /(2M),
(L = Lambda_+), so the trivial azimuthal integration about Q is legitimate; F3 must vanish for a
conserved current (it need not for an incomplete set of diagrams).
"""
import numpy as np
from .dirac import G, G5, C, Cdag, I4, slash
from .quark import sigmas, calF, S as Sq
from .quadrature import radial, chebyshev_u
from .faddeev import basis, dot, SF, TF, _GRP

HBARC = 0.1973269804            # GeV fm
ME_Q = 0.33                     # Euclidean constituent quark mass, GeV (2611.19819 / 2507.13484)
# 2507.13484 Table 1
VERTEX_PARAMS = dict(a=(0.17, 1.47, -0.93), b=(1.03, -0.17, 1.61), r_sc=0.31, r_av=1.05,
                     kappa01=2.64, kappa_phase=-1j)
# kappa_phase: in OUR gamma-matrix conventions (g_{1+} imaginary) the quark loop inside the diquark
# (diquark_triangle / transition_kappas) gives chi^{01} = chi^{10} = (i/M) kappa eps l1 l2 with
# kappa = -2.39 i.  We keep the magnitude 2.64 of 2507.13484 and take phase and the relative sign
# (chi^{10} = +chi^{01}) from the loop; the paper's "chi^{10} = -chi^{01}" refers to its own
# conventions, and taken literally here would make the two transition terms cancel exactly.
EQ = np.diag([2 / 3, -1 / 3])   # quark charges
EB = np.eye(2) / 3              # baryon number per quark

SIGMA = np.array([[0.5j * (G[m] @ G[n] - G[n] @ G[m]) for n in range(4)] for m in range(4)])
EPS = np.zeros((4, 4, 4, 4))
import itertools
for perm in itertools.permutations(range(4)):
    EPS[perm] = np.linalg.det(np.eye(4)[list(perm)])          # eps_1234 = +1

def divdiff(f, x1, x2, h=1e-6):
    """[f(x1) - f(x2)]/(x1 - x2), falling back to a central derivative when x1 ~ x2."""
    x1, x2 = np.asarray(x1, complex), np.asarray(x2, complex)
    d = x1 - x2
    small = np.abs(d) < 1e-9 * (1 + np.abs(x1))
    dd = np.where(small, 1.0, d)
    val = (f(x1) - f(x2)) / dd
    xm = 0.5 * (x1 + x2)
    der = (f(xm + h) - f(xm - h)) / (2 * h)
    return np.where(small, der, val)

def lam_plus(P, M):
    return (-1j * slash(P) + M * I4) / (2 * M)

# ---------------------------------------------------------------- vertices -----
class Vertices:
    def __init__(self, mdq=(0.79, 0.89), M=1.18, params=VERTEX_PARAMS):
        self.ms, self.ma = mdq
        self.ws2, self.wa2 = 0.5 * self.ms**2, 0.5 * self.ma**2
        self.M = M
        self.p = params

    # diquark propagators (scalar functions)
    def D0(self, K2):
        return calF(K2 / self.ws2) / self.ms**2

    def D1(self, K2):
        return calF(K2 / self.wa2) / self.ma**2

    def D1mat(self, K):
        K2 = dot(K, K)
        return (np.eye(4) + K[..., :, None] * K[..., None, :] / self.ma**2) * self.D1(K2)[..., None, None]

    def Fj(self, Q2, r_fm):
        r = r_fm / HBARC
        return 1 / (1 + r**2 * Q2 / 6)

    def s_i(self, Q2):
        E = np.sqrt(1 + Q2 / (2 * ME_Q)**2) - 1
        return [a + b * np.exp(-0.25 * E) for a, b in zip(self.p['a'], self.p['b'])]

    def chi_quark(self, l1, l2):
        """Unamputated quark-photon vertex chi_mu(l1, l2), Eq. (24) of 2507.13484 -> (...,mu,4,4).
        WGTI: i (l1-l2).chi = S(l2) - S(l1)."""
        sV = lambda x: sigmas(x)[0]
        sS = lambda x: sigmas(x)[1]
        a1, a2 = dot(l1, l1), dot(l2, l2)
        SigV = 0.5 * (sV(a1) + sV(a2))
        DV, DS = divdiff(sV, a1, a2), divdiff(sS, a1, a2)
        l, lc = 0.5 * (l1 + l2), l1 - l2
        lc2 = dot(lc, lc)
        s1, s2, s3 = self.s_i(lc2)
        gl, glc = slash(l), slash(lc)
        X = lambda a: np.asarray(a)[..., None, None, None]      # scalar -> (...,mu,4,4)
        chi = G * X(SigV)
        bracket = gl * DV[..., None, None] + 1j * DS[..., None, None] * I4   # (...,4,4)
        chi = chi + 2 * l[..., :, None, None] * bracket[..., None, :, :]
        Gmu = G                                                   # (mu,4,4)
        t1 = glc[..., None, :, :] @ Gmu @ gl[..., None, :, :] - gl[..., None, :, :] @ Gmu @ glc[..., None, :, :]
        chi = chi + 0.5 * X((2 * s1 - 1) * DV) * t1
        sig_lc = np.einsum('mnab,...n->...mab', SIGMA, lc)       # sigma_{mu nu} lc_nu
        chi = chi - X((2 * s2 - 1) * DS) * sig_lc
        chi = chi + 0.5j * X((1 + s3) * DV) * (glc[..., None, :, :] @ sig_lc)
        return chi

    def chi_s(self, l1, l2):
        """Unamputated scalar-diquark vertex (scalar function times index mu): (...,mu)."""
        a1, a2 = dot(l1, l1), dot(l2, l2)
        l, lc = 0.5 * (l1 + l2), l1 - l2
        return -2 * l * (divdiff(self.D0, a1, a2) * self.Fj(dot(lc, lc), self.p['r_sc']))[..., None]

    def chi_a(self, l1, l2, kappa=0.0):
        """Unamputated axial-vector-diquark vertex chi_{mu alpha beta}, Eq. (27): (...,mu,al,be).
        WGTI: lc_mu chi_{mu ab} = F_av [Delta_ab(l2) - Delta_ab(l1)]."""
        a1, a2 = dot(l1, l1), dot(l2, l2)
        l, lc = 0.5 * (l1 + l2), l1 - l2
        Fav = self.Fj(dot(lc, lc), self.p['r_av'])
        dD = divdiff(self.D1, a1, a2)
        m2 = self.ma**2
        d = np.eye(4)
        t = 2 * d[None, :, :] * (l * dD[..., None])[..., :, None, None]
        t = t + (d[:, :, None] * (l1 * self.D1(a1)[..., None])[..., None, None, :]
                 + d[:, None, :] * (l2 * self.D1(a2)[..., None])[..., None, :, None]) / m2
        t = t + 2 / m2 * (l[..., :, None, None] * l2[..., None, :, None] * l1[..., None, None, :]
                          * dD[..., None, None, None])
        chi = -Fav[..., None, None, None] * t
        if kappa:
            chi = chi - kappa * (lc[..., None, None, :] * d[:, :, None] - lc[..., None, :, None] * d[:, None, :]) \
                  * dD[..., None, None, None]
        return chi

    def chi_01(self, l1, l2):
        """Amputated 0+ <-> 1+ transition vertex chi^{01}_{mu alpha} (= chi^{10}, see VERTEX_PARAMS)."""
        lc = l1 - l2
        Fav = self.Fj(dot(lc, lc), self.p['r_av'])
        e = np.einsum('mars,...r,...s->...ma', EPS, l1, l2)
        kappa = self.p['kappa01'] * self.p.get('kappa_phase', 1)
        return 1j / self.M * kappa * e * Fav[..., None, None]

# ----------------------------------------------------------- flavour factors ----
def flavour_factors(charge, phi):
    """Flavour (x charge) factors of diagrams 1-3 for nucleon isospinor phi."""
    f1, f2, f3 = {}, {}, {}
    for X in 'sa':
        f1[X] = sum(phi @ SF[m].T @ charge @ SF[m] @ phi for m in _GRP[X]).real
        f2[X] = sum((phi @ SF[m].T @ SF[n] @ phi) *
                    np.trace(TF[m].T @ (charge @ TF[n] + TF[n] @ charge))
                    for m in _GRP[X] for n in _GRP[X]).real
    for X, Y in (('s', 'a'), ('a', 's')):
        # transition: antisymmetric (Q_1 - Q_2) charge flips the diquark's flavour symmetry
        f3[X + Y] = sum((phi @ SF[m].T @ SF[n] @ phi) *
                        np.trace(TF[m].T @ (charge @ TF[n] - TF[n] @ charge))
                        for m in _GRP[X] for n in _GRP[Y]).real
    return f1, f2, f3

# ---------------------------------------------------------------- current -----
class NucleonCurrent:
    def __init__(self, fad, nk=32, nz=16, ny=12, scale=0.6, eta=1 / 3, params=VERTEX_PARAMS):
        assert fad.M is not None, "solve the Faddeev equation first"
        self.fad, self.M, self.eta = fad, fad.M, eta
        self.V = Vertices((fad.ms, fad.ma), fad.M, params)
        r, wr = radial(nk, scale)
        z, wz = chebyshev_u(nz)
        y, wy = np.polynomial.legendre.leggauss(ny)
        R, Z, Y = np.meshgrid(r, z, y, indexing='ij')
        sz, sy = np.sqrt(1 - Z**2), np.sqrt(1 - Y**2)
        # k = |k| (sz sy, 0, sz y, z): y = cos(angle between spatial k and Q along axis 3)
        self.k = np.stack([R * sz * sy, 0 * R, R * sz * Y, R * Z], -1).reshape(-1, 4)
        self.w = (wr[:, None, None] * r[:, None, None]**3 * wz[None, :, None] * wy[None, None, :]
                  * 2 * np.pi / (2 * np.pi)**4).reshape(-1)

    # nucleon amplitude (including Lambda_+) and its conjugate
    def amplitude(self, l, P):
        f = self.fad.scalar_functions(l, P)                       # (N,8)
        lhat = l / np.sqrt(dot(l, l)).real[..., None]
        Bs, Ba = basis(lhat, P / self.M)
        L = lam_plus(P, self.M)
        Ps = np.einsum('nc,ncab->nab', f[:, :2], Bs) @ L
        Pa = np.einsum('nc,ncvab->nvab', f[:, 2:], Ba) @ L
        return Ps, Pa

    def conj_amplitude(self, p, Pf):
        Ps, Pa = self.amplitude(-p, -Pf)
        Cs = Cdag @ np.swapaxes(Ps, -1, -2) @ C
        Ca = Cdag @ np.swapaxes(Pa, -1, -2) @ C
        return Cs, Ca

    def kinematics(self, Q):
        E = np.sqrt(self.M**2 + Q**2 / 4)
        Pbf = np.array([0, 0, 0, 1j * E])
        Qv = np.array([0, 0, Q, 0], dtype=complex)
        return Pbf + Qv / 2, Pbf - Qv / 2, Qv

    def projectors(self, Pf, Pi, Qv):
        """EM (F1, F2, F3) projection; kept for the normalisation and diagnostics."""
        from .probes import EMProbe
        return EMProbe.projectors(None, self.M, Pf, Pi, Qv)

    def _project(self, Jpt, Tt):
        """Jpt: (N, mu, 4, 4) integrand -> sum_mu Tr[T~_b J_mu] -> (nb, N)."""
        return np.einsum('bmxy,nmyx->bn', Tt, Jpt)

    def diagrams(self, Q, charge=EQ, phi=np.array([1., 0.]), which=(1, 2, 3), probe=None, split=False):
        """Projected form factors of each diagram, un-normalised (divide by the normalisation B).
        probe: EMProbe (default; F1, F2, F3) or GravityProbe (A, J, D, S, cbar).
        split=True returns, per diagram, the parts with scalar ('s') / axial ('a') diquark in
        both nucleons and the mixed part ('m', diagram 3)."""
        if probe is None:
            from .probes import EMProbe
            probe = self._em = getattr(self, '_em', None) or EMProbe((self.fad.ms, self.fad.ma), self.M, self.V.p)
        eta, etb = self.eta, 1 - self.eta
        Pf, Pi, Qv = self.kinematics(Q)
        Tt, Ginv = probe.projectors(self.M, Pf, Pi, Qv)
        f1, f2, f3 = probe.flavour(charge, phi)
        k, w, V = self.k, self.w, self.V
        Ps_i, Pa_i = self.amplitude(k, Pi)
        kq, kd = eta * Pi + k, etb * Pi - k
        out = {}
        I = lambda J: self._integrate(J, Tt, Ginv, w)
        if 1 in which:                       # probe on the bystander quark
            p = k + etb * Qv
            Cs, Ca = self.conj_amplitude(p, Pf)
            pq = eta * Pf + p
            vq = probe.quark(pq, kq)                              # (N,m,4,4)
            Js = f1['s'] * np.einsum('nab,nmbc,ncd->nmad', Cs, vq, Ps_i) * V.D0(dot(kd, kd))[:, None, None, None]
            Ya = np.einsum('nab,nbcd->nacd', V.D1mat(kd), Pa_i)  # Delta_ab Psi_b
            Ja = f1['a'] * np.einsum('naxy,nmyz,nazw->nmxw', Ca, vq, Ya)
            out[1] = {'s': I(Js), 'a': I(Ja)}
        if 2 in which or 3 in which:
            p = k - eta * Qv
            Cs, Ca = self.conj_amplitude(p, Pf)
            pd = etb * Pf - p
            SPs = Sq(kq) @ Ps_i                                   # S(k_q) Psi_s
            SPa = Sq(kq)[:, None] @ Pa_i                          # S(k_q) Psi_a,beta
            if 2 in which:                   # elastic probe-diquark
                c0 = probe.dq_s(pd, kd)                           # (N,m)
                Js = f2['s'] * c0[:, :, None, None] * (Cs @ SPs)[:, None]
                c1 = probe.dq_a(pd, kd)                           # (N,m,al,be)
                Ja = f2['a'] * np.einsum('naxy,nmab,nbyz->nmxz', Ca, c1, SPa)
                out[2] = {'s': I(Js), 'a': I(Ja)}
            if 3 in which:                   # 0+ <-> 1+ transition
                t01 = probe.trans(pd, kd)                         # (N,m,alpha)
                D0p, D0k = V.D0(dot(pd, pd)), V.D0(dot(kd, kd))
                # out scalar, in axial: Psibar_s D0(pd) chi01_{m a} D1_{ab}(kd) S Psi_a,b
                v_sa = np.einsum('nma,nab->nmb', t01, V.D1mat(kd)) * D0p[:, None, None]
                J_sa = f3['sa'] * np.einsum('nxy,nmb,nbyz->nmxz', Cs, v_sa, SPa)
                # out axial, in scalar: Psibar_a,a D1_{ab}(pd) chi10_{m b} D0(kd) S Psi_s
                v_as = np.einsum('nab,nmb->nma', V.D1mat(pd), t01) * D0k[:, None, None]
                J_as = f3['as'] * np.einsum('naxy,nma,nyz->nmxz', Ca, v_as, SPs)
                out[3] = {'m': I(J_sa + J_as)}
        if split:
            return out
        return {d: sum(v.values()) for d, v in out.items()}

    def _integrate(self, Jpt, Tt, Ginv, w):
        return Ginv @ (self._project(Jpt, Tt) @ w)

    # ---------- normalisation from the Faddeev eigenvalue (Ward identity) ----------
    def psibar_G0_psi(self, M=None):
        """n_0 with int Psibar G0 Psi = n_0 Lambda_+ (rest frame), and the P-derivative piece
        -Psibar dG0/dP_mu Psi projected on T1 (= diagrams 1+2 with baryon-number charges at Q=0)."""
        M = self.M
        P = np.array([0, 0, 0, 1j * M])
        eta, etb = self.eta, 1 - self.eta
        k, w, V = self.k, self.w, self.V
        Ps, Pa = self.amplitude(k, P)
        Cs, Ca = self.conj_amplitude(k, P)
        def G0sand(Pt):
            kq, kd = eta * Pt + k, etb * Pt - k
            S = Sq(kq)
            js = Cs @ S @ Ps * V.D0(dot(kd, kd))[:, None, None]
            ja = np.einsum('naxy,nyz,nab,nbzw->nxw', Ca, S, V.D1mat(kd), Pa)
            return js + ja                                         # (N,4,4)
        n0 = (np.einsum('nxx->n', G0sand(P)) @ w) / 2              # Tr Lambda_+ = 2
        h = 1e-4
        dJ = np.zeros((len(w), 4, 4, 4), complex)
        for mu in range(4):
            e = np.zeros(4, complex); e[mu] = h
            dJ[:, mu] = -(G0sand(P + e) - G0sand(P - e)) / (2 * h)
        Tt, Ginv = self.projectors(P, P, np.zeros(4, complex))
        F1_G0 = self._integrate(dJ, Tt, Ginv, w)[0]
        return n0, F1_G0

    def normalisation(self, dM=1e-3):
        """Full (all six diagrams) baryon-number charge of the un-normalised amplitude:
        J^B_mu(0) = -dlambda/dP_mu  int Psibar G0 Psi  =  (P_mu/M)(dlambda/dM) n_0 Lambda_+,
        projected on T1.  Dividing by it normalises the Faddeev amplitude canonically."""
        lp = self.fad.eigen(self.M + dM)[0]
        lm = self.fad.eigen(self.M - dM)[0]
        dlam = (lp - lm) / (2 * dM)
        n0, F1_G0 = self.psibar_G0_psi()
        P = np.array([0, 0, 0, 1j * self.M])
        Tt, Ginv = self.projectors(P, P, np.zeros(4, complex))
        L = lam_plus(P, self.M)
        JB = (P / self.M)[:, None, None] * dlam * n0 * L           # (mu,4,4)
        B = (Ginv @ np.einsum('bmxy,myx->b', Tt, JB))[0]
        return B, F1_G0, dlam, n0

# ------------------------------------------------ diquark-photon quark loop -----
def diquark_triangle(X, Y, l1, l2, g, mdq=(0.79, 0.89), nq=48, nz=24, ny=16, params=VERTEX_PARAMS):
    """Photon coupling to the quarks inside a diquark, computed as a quark loop with our own
    diquark amplitudes (same conventions as the Faddeev kernel and the canonical normalisation):
        T_a = int Tr[ Gbar^X(q+Q/2; -l1) i chi_mu(q_a+Q, q_a) G^Y(q; l2) S^T(q_b) ]       (photon on a)
        T_b = int Tr[ Gbar^X(q-Q/2; -l1) S(q_a) G^Y(q; l2) (i chi_mu(q_b+Q, q_b))^T ]     (photon on b)
    q_a = q + l2/2, q_b = -q + l2/2, Q = l1 - l2, colour trace 2, flavour stripped.
    Returns T_a, T_b with shape (mu,) [00], (mu, alpha) [01: alpha on G^Y; 10: alpha on Gbar^X]."""
    from .faddeev import GAM_S, GAM_A, BAR_S, BAR_A
    V = Vertices(mdq, 1.0, params)
    w2 = {'s': 0.5 * mdq[0]**2, 'a': 0.5 * mdq[1]**2}
    gg = {'s': g[0], 'a': g[1]}
    r, wr = radial(nq, 1.0)
    z, wz = chebyshev_u(nz)
    y, wy = np.polynomial.legendre.leggauss(ny)
    R, Z, Yy = np.meshgrid(r, z, y, indexing='ij')
    sz, sy = np.sqrt(1 - Z**2), np.sqrt(1 - Yy**2)
    q = np.stack([R * sz * sy, 0 * R, R * sz * Yy, R * Z], -1).reshape(-1, 4)
    w = (wr[:, None, None] * r[:, None, None]**3 * wz[None, :, None] * wy[None, None, :]
         * 2 * np.pi / (2 * np.pi)**4).reshape(-1)
    Q = l1 - l2
    qa, qb = q + l2 / 2, -q + l2 / 2
    def amp(T, rel, bar):
        F = calF(dot(rel, rel) / w2[T]) * gg[T]
        D = (BAR_S[None] if T == 's' else BAR_A) if bar else (GAM_S[None] if T == 's' else GAM_A)
        return F[:, None, None, None] * D[None]                      # (N, idx, 4, 4)
    Gy = amp(Y, q, False)
    out = []
    for side in 'ab':
        if side == 'a':
            Gx = amp(X, q + Q / 2, True)
            ins = 1j * V.chi_quark(qa + Q, qa)                       # (N,mu,4,4)
            other = np.swapaxes(Sq(qb), -1, -2)
            # Tr[Gx_i ins_mu Gy_j other]
            val = np.einsum('nixy,nmyz,njzw,nwx->nmij', Gx, ins, Gy, other)
        else:
            Gx = amp(X, q - Q / 2, True)
            ins = np.swapaxes(1j * V.chi_quark(qb + Q, qb), -1, -2)
            val = np.einsum('nixy,nyz,njzw,nmwx->nmij', Gx, Sq(qa), Gy, ins)
        out.append(2 * np.einsum('n,nmij->mij', w, val))             # colour trace 2
    shape = {('s', 's'): lambda t: t[:, 0, 0], ('s', 'a'): lambda t: t[:, 0, :],
             ('a', 's'): lambda t: t[:, :, 0]}[(X, Y)]
    return shape(out[0]), shape(out[1])

def transition_kappas(g, mdq=(0.79, 0.89), M=1.18, Qs=0.1, **kw):
    """Effective kappa_01, kappa_10 (in chi = (i/M) kappa eps_{mu al rho la} l1_rho l2_la) implied
    by the quark loop in OUR conventions: physical amputated vertex = -(T_a - T_b)/2 per unit
    (Q_1 - Q_2) charge, where the sign -1 is fixed by the elastic scalar-diquark Ward identity.
    Evaluated at l1^2 = l2^2 = -mbar^2, |Q| = Qs (Breit-like)."""
    mbar = 0.5 * sum(mdq)
    E = np.sqrt(mbar**2 + Qs**2 / 4)
    l1 = np.array([0, 0, Qs / 2, 1j * E]); l2 = np.array([0, 0, -Qs / 2, 1j * E])
    eps = np.einsum('mars,r,s->ma', EPS, l1, l2)
    norm = np.einsum('ma,ma->', eps, eps)
    res = {}
    for X, Y in (('s', 'a'), ('a', 's')):
        Ta, Tb = diquark_triangle(X, Y, l1, l2, g, mdq, **kw)
        # flavour: vertex = Tr[t^T Q t] T_a + Tr[t^T t Q] T_b = Tr[t^T (Qt - tQ)] (T_a - T_b)/2
        #          provided T_b = -T_a (checked by the returned ratio)
        phys = -(Ta - Tb) / 2
        c = np.einsum('ma,ma->', phys, eps) / norm
        res[X + Y] = dict(kappa=c * M / 1j, asym=np.abs(Ta + Tb).max() / np.abs(Ta).max())
    return res
