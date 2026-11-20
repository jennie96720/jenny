"""Nucleon quark+diquark Faddeev equation (scalar + axial-vector diquarks).

Kernel: Alkofer, Höll, Kloker, Krassnigg, Roberts, nucl-th/0412046, Eqs. (8)-(13)
(identical to Cloët et al. arXiv:0812.0416 App. A), written in the colour-flavour
conventions of arXiv:2611.19819 (flavour matrices t_f, s_f with unit norm):

  psi_X(k) = -4 sum_Y c_XY g_X g_Y  int d^4l/(2pi)^4
             Gamma^Y(k_q - l_qq/2) S^T(l_qq - k_q) Gammabar^X(l_q - k_qq/2) S(l_q) Delta^Y(l_qq) psi_Y(l)

  X, Y in {s (0+), a (1+)};  l_q = l + P/3,  l_qq = -l + 2P/3  (same for k);
  -4 = 2 (Psi_3 couples to Psi_1 and Psi_2) x (-2) (colour contraction H^a H^a = -2);
  c_XY = sum_{m in X, n in Y} phi^T (s^m)^T t^n (t^m)^T s^n phi   (App. B of 2611.19819).

The Dirac part of each amplitude is expanded in the 2 + 6 positive-energy structures
S^{1,2}, gamma_5 A^{1..6}_nu of 2611.19819 Eq. (A15), times Lambda_+(P).
Rest frame: P = (0,0,0, i M).  Nystrom discretisation on (|l|, z = l_4/|l|);
the angle y between the 3-vector parts of k and l is integrated with Gauss-Legendre,
the remaining azimuth trivially (2 pi).
"""
import numpy as np
from .dirac import G, G5, C, Cdag, I4, slash
from .quark import S as Sq, calF
from .quadrature import radial, chebyshev_u

# ---- flavour coefficients -------------------------------------------------
_t = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1., -1.])]
TF = [1j / np.sqrt(2) * _t[2], 0.5 * (_t[0] + _t[3]), _t[1] / np.sqrt(2), 0.5 * (_t[0] - _t[3])]
SF = [_t[0], (_t[1] - 1j * _t[2]) / np.sqrt(6), -_t[3] / np.sqrt(3), -(_t[1] + 1j * _t[2]) / np.sqrt(6)]
_GRP = {'s': [0], 'a': [1, 2, 3]}

def flavour_coeffs(phi=np.array([1., 0.])):
    """c_XY for the proton (phi_+) or neutron (phi_-); identical by isospin symmetry."""
    return {X + Y: sum(phi @ SF[m].T @ TF[n] @ TF[m].T @ SF[n] @ phi
                       for m in _GRP[X] for n in _GRP[Y]).real
            for X in 'sa' for Y in 'sa'}

# ---- Dirac structures of the diquark amplitudes (Alkofer et al. Eq. 15) ----
GAM_S = C @ (1j * G5)                                     # Gamma^{0+} ~ C i gamma_5
GAM_A = np.array([1j * G[m] @ C for m in range(4)])       # Gamma^{1+}_mu ~ i gamma_mu C
BAR_S = Cdag @ GAM_S.T @ C                                # Gammabar = C^dag Gamma^T C
BAR_A = np.array([Cdag @ GAM_A[m].T @ C for m in range(4)])

def dot(a, b):
    return np.einsum('...m,...m->...', a, b)

# ---- nucleon amplitude basis (2611.19819 Eq. A15) ------------------------
def basis(lhat, Phat):
    """lhat: (...,4) real unit vectors; Phat = (0,0,0,i).
    Returns Bs (...,2,4,4) and Ba (...,6,4,4,4) [n, nu, Dirac, Dirac] with gamma_5 included."""
    lP = dot(lhat, Phat)[..., None]                        # = i z
    lperp = lhat + lP * Phat                               # spatial part
    lperp = lperp / np.sqrt(dot(lperp, lperp))[..., None]  # unit: lperp.lperp = 1
    gl, glp, gP = slash(lhat), slash(lperp), slash(np.broadcast_to(Phat, lhat.shape))
    sh = lhat.shape[:-1]
    Bs = np.empty(sh + (2, 4, 4), complex)
    Bs[..., 0, :, :] = I4
    Bs[..., 1, :, :] = 1j * gl - lP[..., None] * I4
    gperp = G + gP[..., None, :, :] * Phat[:, None, None]  # gamma^perp_nu: (...,nu,4,4)
    A = np.empty(sh + (6, 4, 4, 4), complex)
    A[..., 0, :, :, :] = glp[..., None, :, :] * Phat[:, None, None]
    A[..., 1, :, :, :] = -1j * Phat[:, None, None] * I4
    A[..., 2, :, :, :] = glp[..., None, :, :] * lperp[..., :, None, None]
    A[..., 3, :, :, :] = 1j * lperp[..., :, None, None] * I4
    A[..., 4, :, :, :] = gperp - A[..., 2, :, :, :]
    A[..., 5, :, :, :] = 1j * gperp @ glp[..., None, :, :] - A[..., 3, :, :, :]
    Ba = np.einsum('ab,...nvbc->...nvac', G5, A)
    return Bs, Ba

class NucleonFaddeev:
    def __init__(self, g=(14.94, 12.98), mdq=(0.79, 0.89), npts=(20, 10, 8),
                 scale=0.6, channels='sa'):
        self.gs, self.ga = g
        self.ms, self.ma = mdq
        self.ws2, self.wa2 = 0.5 * self.ms**2, 0.5 * self.ma**2
        self.channels = channels
        self.c = flavour_coeffs()
        nr, nz, ny = npts
        self.r, self.wr = radial(nr, scale)
        self.z, self.wz = chebyshev_u(nz)
        self.y, self.wy = np.polynomial.legendre.leggauss(ny)
        self.nr, self.nz, self.ny = nr, nz, ny
        # external grid = Nystrom nodes: k = r (0, 0, sqrt(1-z^2), z)
        R, Z = np.meshgrid(self.r, self.z, indexing='ij')
        self.kvec = np.stack([0 * R, 0 * R, R * np.sqrt(1 - Z**2), R * Z], -1)   # (nr,nz,4)
        # integration grid: l = r (0, s_z s_y, s_z y, z)
        R3, Z3, Y3 = np.meshgrid(self.r, self.z, self.y, indexing='ij')
        sz, sy = np.sqrt(1 - Z3**2), np.sqrt(1 - Y3**2)
        self.lvec = np.stack([0 * R3, R3 * sz * sy, R3 * sz * Y3, R3 * Z3], -1)  # (nr,nz,ny,4)
        self.wl = (self.wr[:, None, None] * self.r[:, None, None]**3 * self.wz[None, :, None]
                   * self.wy[None, None, :] * 2 * np.pi / (2 * np.pi)**4)
        self.Phat = np.array([0, 0, 0, 1j])
        # amplitude bases at the integration and external points
        lhat = self.lvec / np.linalg.norm(self.lvec, axis=-1, keepdims=True)
        self.Bs_l, self.Ba_l = basis(lhat, self.Phat)
        khat = self.kvec / np.linalg.norm(self.kvec, axis=-1, keepdims=True)
        Bs_k, Ba_k = basis(khat, self.Phat)
        Lp = 0.5 * (I4 + G[3])                    # Lambda_+ for P = (0,0,0,iM)
        self.Lp = Lp
        self.Ds, self.Da = self._duals(Bs_k @ Lp, Ba_k @ Lp)

    @staticmethod
    def _duals(Bs, Ba):
        """Dual bases D_m with sum_mu Tr[D_m R] = coefficient m of R (least-squares projection)."""
        # scalar channel
        Gs = np.einsum('...mab,...nab->...mn', Bs.conj(), Bs)
        Ds = np.einsum('...mn,...nab->...mba', np.linalg.inv(Gs), Bs.conj())
        Ga = np.einsum('...mvab,...nvab->...mn', Ba.conj(), Ba)
        Da = np.einsum('...mn,...nvab->...mvba', np.linalg.inv(Ga), Ba.conj())
        return Ds, Da

    def kernel_matrix(self, M):
        """Return K with psi = K psi, unknowns ordered (channel n=0..7, r, z)."""
        P = np.array([0, 0, 0, 1j * M])
        nr, nz = self.nr, self.nz
        Npt = nr * nz
        use_s, use_a = 's' in self.channels, 'a' in self.channels
        l = self.lvec
        lq = l + P / 3
        lqq = -l + 2 * P / 3
        S_lq = Sq(lq)                                               # (nr,nz,ny,4,4)
        Fdq_s = calF(dot(lqq, lqq) / self.ws2) / self.ms**2         # Pi^{0+}(l_qq^2)
        Fdq_a = calF(dot(lqq, lqq) / self.wa2) / self.ma**2
        Dlt_a = (np.eye(4) + lqq[..., :, None] * lqq[..., None, :] / self.ma**2) * Fdq_a[..., None, None]
        # psi basis times Lambda_+ at the integration points
        Bs = self.Bs_l @ self.Lp                                   # (nr,nz,ny,2,4,4)
        Ba = self.Ba_l @ self.Lp                                   # (nr,nz,ny,6,4,4,4)
        # incoming-diquark side contracted with psi:  Delta^Y psi_Y
        #   scalar: Delta^0 * Bs_n;  axial: sum_nu Delta_{rho nu} Ba_{n,nu} -> (n, rho)
        inS = Fdq_s[..., None, None, None] * Bs
        inA = np.einsum('...rv,...nvab->...nrab', Dlt_a, Ba)
        K = np.zeros((8, nr, nz, 8, nr, nz), complex)
        w = self.wl
        for i in range(nr):
            for a in range(nz):
                k = self.kvec[i, a]
                kq, kqq = k + P / 3, -k + 2 * P / 3
                rel_in = kq - lqq / 2           # argument of Gamma^Y
                rel_out = lq - kqq / 2          # argument of Gammabar^X
                SexT = np.swapaxes(Sq(lqq - kq), -1, -2)
                Fin_s = calF(dot(rel_in, rel_in) / self.ws2)
                Fin_a = calF(dot(rel_in, rel_in) / self.wa2)
                Fout_s = calF(dot(rel_out, rel_out) / self.ws2)
                Fout_a = calF(dot(rel_out, rel_out) / self.wa2)
                # middle = S^T(ex) Gammabar^X S(l_q)
                mid_s = SexT @ BAR_S @ S_lq                              # (..,4,4)
                mid_a = np.einsum('...ab,mbc,...cd->...mad', SexT, BAR_A, S_lq)  # (..,mu,4,4)
                Ds, Da = self.Ds[i, a], self.Da[i, a]
                blocks = []
                if use_s:   # X = s (outgoing scalar)
                    if use_s:   # Y = s
                        T = np.einsum('ab,...bc,...ncd->...nad', GAM_S, mid_s, inS)
                        coef = -4 * self.c['ss'] * self.gs**2 * Fin_s * Fout_s
                        blocks.append((0, 0, np.einsum('mba,...nab->...mn', Ds, T), coef))
                    if use_a:   # Y = a
                        T = np.einsum('rab,...bc,...nrcd->...nad', GAM_A, mid_s, inA)
                        coef = -4 * self.c['sa'] * self.gs * self.ga * Fin_a * Fout_s
                        blocks.append((0, 2, np.einsum('mba,...nab->...mn', Ds, T), coef))
                if use_a:   # X = a (outgoing axial, free index mu)
                    if use_s:
                        T = np.einsum('ab,...mbc,...ncd->...nmad', GAM_S, mid_a, inS)
                        coef = -4 * self.c['as'] * self.ga * self.gs * Fin_s * Fout_a
                        blocks.append((2, 0, np.einsum('kmba,...nmab->...kn', Da, T), coef))
                    if use_a:
                        T = np.einsum('rab,...mbc,...nrcd->...nmad', GAM_A, mid_a, inA)
                        coef = -4 * self.c['aa'] * self.ga**2 * Fin_a * Fout_a
                        blocks.append((2, 2, np.einsum('kmba,...nmab->...kn', Da, T), coef))
                for (o_off, i_off, proj, coef) in blocks:
                    # proj: (nr,nz,ny, n_out, n_in); integrate over y here
                    val = np.einsum('rzy,rzymn->mnrz', w * coef, proj)
                    no, ni = val.shape[:2]
                    K[o_off:o_off + no, i, a, i_off:i_off + ni] += val
        idx = [n for n in range(8) if (n < 2 and use_s) or (n >= 2 and use_a)]
        K = K[idx][:, :, :, idx]
        n = len(idx) * Npt
        return K.reshape(n, n)

    def eigen(self, M):
        """Eigenvalue with the largest real part (the physical ground state has lambda = 1)."""
        ev, evec = np.linalg.eig(self.kernel_matrix(M))
        j = np.argmax(ev.real)
        return ev[j], evec[:, j]

    def solve_mass(self, M0=1.1, M1=1.3, tol=1e-5, verbose=True):
        """Secant search for lambda(M) = 1."""
        f0 = self.eigen(M0)[0].real - 1
        f1 = self.eigen(M1)[0].real - 1
        for _ in range(30):
            M2 = M1 - f1 * (M1 - M0) / (f1 - f0)
            M0, f0 = M1, f1
            M1, f1 = M2, self.eigen(M2)[0].real - 1
            if verbose:
                print(f"   M = {M1:.5f} GeV   lambda - 1 = {f1:+.2e}")
            if abs(f1) < tol:
                break
        return M1
