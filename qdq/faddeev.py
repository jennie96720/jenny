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
Rest frame: P = (0,0,0, i M), so l.Phat = i z with z = l_4/|l| (cosine of the 4-angle).
Each of the 8 scalar functions is expanded in Chebyshev polynomials of the 2nd kind,
    f_c(l^2, z) = sum_{n < N_cheb} f_{c,n}(|l|) U_n(z),
which (i) reduces the number of unknowns and of external points, and (ii) gives an
analytic continuation to complex z, needed when the amplitude is evaluated with a
boosted (Breit-frame) P in form-factor integrals.  The integration over l uses
Gauss-Chebyshev nodes in z, Gauss-Legendre in the angle y between the 3-vector parts of
k and l, and the remaining azimuth trivially (2 pi).  ncheb=None reproduces the plain
Nystrom discretisation on the (|l|, z) nodes.
"""
import numpy as np
from .dirac import G, G5, C, Cdag, I4, slash
from .quark import S as Sq, calF
from .quadrature import radial, chebyshev_u
from scipy.interpolate import CubicSpline

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

def cheb_u(nmax, z):
    """U_0..U_{nmax-1} at (complex) z -> shape (nmax,) + z.shape."""
    z = np.asarray(z, dtype=complex)
    U = np.empty((nmax,) + z.shape, complex)
    U[0] = 1
    if nmax > 1:
        U[1] = 2 * z
    for n in range(2, nmax):
        U[n] = 2 * z * U[n - 1] - U[n - 2]
    return U

NAMES = ['s1', 's2', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']

class NucleonFaddeev:
    """Nucleon Faddeev equation.  Unknowns f[c, r_i, n]: channel c (s1,s2,a1..a6),
    radial node r_i, Chebyshev moment n (or z-node index if ncheb is None)."""

    def __init__(self, g=(7.47, 6.49j), mdq=(0.79, 0.89), nr=16, nz=8, ny=8, ncheb=4,
                 scale=0.6, channels='sa'):
        self.gs, self.ga = g
        self.ms, self.ma = mdq
        self.ws2, self.wa2 = 0.5 * self.ms**2, 0.5 * self.ma**2
        self.channels = channels
        self.c = flavour_coeffs()
        self.r, self.wr = radial(nr, scale)
        self.z, self.wz = chebyshev_u(nz)
        self.y, self.wy = np.polynomial.legendre.leggauss(ny)
        self.nr, self.nz, self.ny = nr, nz, ny
        self.ncheb = ncheb
        if ncheb is None:                     # Nystrom: external z = integration z
            self.zext, wext = self.z, self.wz
            self.Pout = np.eye(nz)            # node values -> unknowns
            self.Ein = np.eye(nz)             # unknowns -> node values
        else:
            assert ncheb <= nz
            self.zext, wext = chebyshev_u(ncheb)
            self.Pout = 2 / np.pi * wext[None, :] * cheb_u(ncheb, self.zext).real   # (n, a)
            self.Ein = cheb_u(ncheb, self.z).real.T                              # (b, n)
        self.nzext = len(self.zext)
        # external points k = r (0, 0, sqrt(1-z^2), z)
        R, Z = np.meshgrid(self.r, self.zext, indexing='ij')
        self.kvec = np.stack([0 * R, 0 * R, R * np.sqrt(1 - Z**2), R * Z], -1)
        # integration points l = r (0, s_z s_y, s_z y, z)
        R3, Z3, Y3 = np.meshgrid(self.r, self.z, self.y, indexing='ij')
        sz, sy = np.sqrt(1 - Z3**2), np.sqrt(1 - Y3**2)
        self.lvec = np.stack([0 * R3, R3 * sz * sy, R3 * sz * Y3, R3 * Z3], -1).reshape(-1, 4)
        self.wl = (self.wr[:, None, None] * self.r[:, None, None]**3 * self.wz[None, :, None]
                   * self.wy[None, None, :] * 2 * np.pi / (2 * np.pi)**4).reshape(-1)
        self.Phat = np.array([0, 0, 0, 1j])
        self.Lp = 0.5 * (I4 + G[3])           # Lambda_+ for P = (0,0,0,iM)
        lhat = self.lvec / np.linalg.norm(self.lvec, axis=-1, keepdims=True)
        Bs_l, Ba_l = basis(lhat, self.Phat)
        self.Bs_l, self.Ba_l = Bs_l @ self.Lp, Ba_l @ self.Lp
        khat = self.kvec / np.linalg.norm(self.kvec, axis=-1, keepdims=True)
        Bs_k, Ba_k = basis(khat, self.Phat)
        self.Ds, self.Da = self._duals(Bs_k @ self.Lp, Ba_k @ self.Lp)
        self.M = None

    @staticmethod
    def _duals(Bs, Ba):
        """Dual bases D_m (Dirac-transposed for Tr[D R] = sum D[b,a] R[a,b]):
        sum_nu Tr[D_m R_nu] = coefficient m of R in the basis (least-squares projection)."""
        Gs = np.einsum('...mab,...nab->...mn', Bs.conj(), Bs)
        Ds = np.einsum('...mn,...nab->...mab', np.linalg.inv(Gs), Bs.conj())
        Ga = np.einsum('...mvab,...nvab->...mn', Ba.conj(), Ba)
        Da = np.einsum('...mn,...nvab->...mvab', np.linalg.inv(Ga), Ba.conj())
        # coefficient = sum_{nu,a,b} D[m,nu,a,b] R[nu,a,b]
        return Ds, Da

    def _blocks(self):
        use_s, use_a = 's' in self.channels, 'a' in self.channels
        out = []
        for X, ok_x in (('s', use_s), ('a', use_a)):
            for Y, ok_y in (('s', use_s), ('a', use_a)):
                if ok_x and ok_y:
                    out.append((X, Y))
        return out

    def kernel_matrix(self, M):
        """K with f = K f, unknowns ordered (channel c, r_i, n)."""
        P = np.array([0, 0, 0, 1j * M])
        l = self.lvec
        Ni = l.shape[0]
        lq, lqq = l + P / 3, -l + 2 * P / 3
        S_lq = Sq(lq)                                             # (Ni,4,4)
        lqq2 = dot(lqq, lqq)
        Pi_s = calF(lqq2 / self.ws2) / self.ms**2
        Pi_a = calF(lqq2 / self.wa2) / self.ma**2
        Dlt_a = (np.eye(4) + lqq[:, :, None] * lqq[:, None, :] / self.ma**2) * Pi_a[:, None, None]
        # incoming side  Delta^Y psi_Y :  in_s (Ni,2,1,4,4), in_a (Ni,6,4rho,4,4)
        inp = {'s': (Pi_s[:, None, None, None] * self.Bs_l)[:, :, None],
               'a': np.einsum('irv,invab->inrab', Dlt_a, self.Ba_l)}
        bar = {'s': BAR_S[None], 'a': BAR_A}                      # (mu,4,4)
        gam = {'s': GAM_S[None], 'a': GAM_A}                      # (rho,4,4)
        g = {'s': self.gs, 'a': self.ga}
        nb = {'s': 2, 'a': 6}
        # Rin[X,Y] = Gammabar^X_mu S(l_q) in_Y[n,rho] : (Ni, nY, mu, rho, 4, 4)  -- inner only
        Rin = {}
        for X, Y in self._blocks():
            t = np.einsum('mab,ibc->imac', bar[X], S_lq)            # (Ni,mu,4,4)
            R = np.matmul(t[:, None, :, None], inp[Y][:, :, None, :])
            # rows (i, n, Rin row index), cols (mu, rho, Rin column index)
            Rin[X, Y] = R.transpose(0, 1, 4, 2, 3, 5).reshape(Ni * R.shape[1] * 4, -1)
        off = {'s': 0, 'a': 2} if 's' in self.channels else {'a': 0}
        nch = sum(nb[c] for c in off)
        nr, nza, nz, ny = self.nr, self.nzext, self.nz, self.ny
        Kfull = np.zeros((nch, nr, nza, nch, nr * nz), complex)
        for i in range(nr):
            for a in range(nza):
                k = self.kvec[i, a]
                kq, kqq = k + P / 3, -k + 2 * P / 3
                rel_in, rel_out = kq - lqq / 2, lq - kqq / 2
                SexT = np.swapaxes(Sq(lqq - kq), -1, -2)             # (Ni,4,4)
                r_in, r_out = dot(rel_in, rel_in), dot(rel_out, rel_out)
                F = {('in', 's'): calF(r_in / self.ws2), ('in', 'a'): calF(r_in / self.wa2),
                     ('out', 's'): calF(r_out / self.ws2), ('out', 'a'): calF(r_out / self.wa2)}
                D = {'s': self.Ds[i, a][:, None], 'a': self.Da[i, a]}   # (nX, mu, 4, 4)
                for X, Y in self._blocks():
                    # Lout[k, mu, rho] = D_{k,mu} Gamma^Y_rho   (D stored so that coeff = sum D*R)
                    # coefficient = Tr[Dt_k Gamma^Y SexT Rin]  with Dt = D^T
                    Lout = np.einsum('kmba,rbc->kmrac', D[X], gam[Y])     # Dt @ Gamma
                    nX, nY = nb[X], nb[Y]
                    # Tr[Lt SexT R] = sum Lt[x,y] SexT[y,z] R[z,x]; contract (mu, rho, x) by BLAS
                    B1 = Lout.transpose(1, 2, 3, 0, 4).reshape(-1, nX * 4)
                    W = (Rin[X, Y] @ B1).reshape(Ni, nY, 4, nX, 4)          # [i, n, z, k, y]
                    val = np.einsum('iyz,inzky->ikn', SexT, W)
                    coef = (-4 * self.c[X + Y] * g[X] * g[Y] * F['in', Y] * F['out', X] * self.wl)
                    val = (coef[:, None, None] * val).reshape(nr * nz, ny, nX, nY).sum(1)
                    Kfull[off[X]:off[X] + nX, i, a, off[Y]:off[Y] + nY] += val.transpose(1, 2, 0)
        # external z-nodes -> moments, inner (r,z)-nodes <- moments
        K = Kfull.reshape(nch, nr, nza, nch, nr, nz)
        K = np.einsum('ma,ciadjb,bn->cimdjn', self.Pout, K, self.Ein, optimize=True)
        nm = self.Pout.shape[0]
        self._nch, self._nm = nch, nm
        return K.reshape(nch * nr * nm, nch * nr * nm)

    def eigen(self, M):
        """Eigenvalue with the largest real part (ground state: lambda = 1) and its eigenvector."""
        ev, evec = np.linalg.eig(self.kernel_matrix(M))
        j = np.argmax(ev.real)
        return ev[j], evec[:, j]

    def solve_mass(self, M0=1.15, M1=1.20, tol=1e-6, verbose=True):
        """Secant search for lambda(M) = 1; stores the mass and the amplitude moments."""
        f0 = self.eigen(M0)[0].real - 1
        f1 = self.eigen(M1)[0].real - 1
        for _ in range(30):
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
        """Save amplitude moments f[c, n, r] (8 channels; zero for switched-off channels),
        phase/scale fixed by s1 moment 0 at the first radial node = 1."""
        f = vec.reshape(self._nch, self.nr, self._nm)
        full = np.zeros((8, self.nr, self._nm), complex)
        if 's' in self.channels:
            full[:self._nch] = f
        else:
            full[2:] = f
        if self.ncheb is None:              # convert node values to moments for interpolation
            Pm = 2 / np.pi * self.wz[None, :] * cheb_u(self.nz, self.z).real
            full = np.einsum('na,cia->cin', Pm, full)
        ref = full[0, 0, 0] if 's' in self.channels else full[2, 0, 0]
        self.f = (full / ref).transpose(0, 2, 1)          # (c, n, r)
        self.splines = CubicSpline(np.log(self.r), self.f, axis=2)

    def scalar_functions(self, l, P):
        """The 8 functions f_c(l^2, l.P) at arbitrary l (...,4) and on-shell P (4,), P^2 = -M^2.
        Radial dependence: cubic spline in log|l| (constant/power continuation outside grid);
        angular: Chebyshev series, valid also for complex z = -i l.Phat/|l|."""
        l = np.asarray(l, dtype=complex)
        Phat = P / self.M
        L = np.sqrt(dot(l, l))
        z = -1j * dot(l, Phat) / L
        logr = np.clip(np.log(L.real), np.log(self.r[0]), np.log(self.r[-1]))
        fn = self.splines(logr)                          # (c, n, ...)
        fn = np.moveaxis(fn, (0, 1), (-2, -1))           # (..., c, n)
        U = np.moveaxis(cheb_u(fn.shape[-1], z), 0, -1)  # (..., n)
        return np.einsum('...cn,...n->...c', fn, U)
