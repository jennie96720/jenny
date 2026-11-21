"""Quark-in-diquark distribution functions at the hadron scale (PDF step 2).

Prescription of EPJC 84 (2024) 739 App. B and arXiv:2505.10663 App. A (Eqs. A.1-A.5):
    q^{D}(z) = n_D [phi_D(z)]^2,   int_0^1 dz q^D(z) = 1,
with phi_D the leading-twist distribution amplitude of the diquark correlation,
    0+:   f phi(x)       = N tr_D int d^4k/(2pi)^4 delta(n.k_+ - x n.P) gamma_5 gamma.n chi(k;P),
    1+:   n.P f phi(x)   = m N tr_D int d^4k/(2pi)^4 delta(n.k_+ - x n.P) gamma.n n_nu chi_nu(k;P),
    chi(k;P) = S(k_+) Gamma(k;P) C^dagger S(k_-),   k_+- = k +- P/2,
using the diquark amplitudes of qdq/diquark.py (C i gamma_5 F(k^2/w0^2), i gamma_mu C F(k^2/w1^2)) and
the algebraic quark propagator.  Normalisations (f, N, couplings) drop out.

Numerics.  phi is accessed through its Euclidean Mellin moments <x^m>_phi (light-like n, rest frame,
on-shell P^2 = -m_D^2), as for the nucleon (qdq/pdf.py); phi(x) is then reconstructed with the
two-parameter-family form of Chang et al., PRL 110 (2013) 132001,
    phi(x) = N_alpha [x(1-x)]^(alpha - 1/2) [1 + a_2 C_2^alpha(2x - 1) + a_4 C_4^alpha(2x - 1)],
fitted to <xi^2>, <xi^4>, <xi^6> (xi = 2x - 1; reliable to <1e-3 with kmax = 15-30 GeV).  The
two-parameter form (a_4 = 0) cannot reach the 1+ moments (<xi^4>/<xi^2>^2 too large for its width);
the spread between the (2,4) and (2,4,6) fits is a systematic error estimate.  Finally <z^m>_{q/D} = moments
of phi^2 (exact quadrature).
"""
import numpy as np
from scipy.special import gegenbauer, gamma as Gf, beta as Bf
from scipy.optimize import least_squares
from .dirac import G, G5, C, Cdag, I4, slash
from .quark import S as Sq, calF
from .quadrature import radial, chebyshev_u
from .diquark import MASS, omega2

NV = np.array([0, 0, 1, 1j])

def loop4d_full(nq, nz, ny, scale=1.0, kmax=None):
    """int d^4k/(2pi)^4 for integrands depending on (|k|, k_4, k_3):
    k = |k| (0, sqrt(1-z^2) sqrt(1-y^2), sqrt(1-z^2) y, z); the last azimuth gives 2 pi.
    Gauss-Chebyshev (2nd kind) in z, Gauss-Legendre in y.  Returns (N, 4) vectors and (N,) weights."""
    q, wq = radial(nq, scale)
    if kmax is not None:
        keep = q < kmax
        q, wq = q[keep], wq[keep]
    z, wz = chebyshev_u(nz)
    y, wy = np.polynomial.legendre.leggauss(ny)
    Q, Z, Y = np.meshgrid(q, z, y, indexing='ij')
    W = (wq[:, None, None] * q[:, None, None]**3) * wz[None, :, None] * wy[None, None, :] \
        * 2 * np.pi / (2 * np.pi)**4
    st = np.sqrt(1 - Z**2)
    vec = np.stack([np.zeros_like(Q), Q * st * np.sqrt(1 - Y**2), Q * st * Y, Q * Z], axis=-1)
    return vec.reshape(-1, 4), W.reshape(-1)

def da_moments(ch, mmax=8, nq=128, nz=64, ny=14, scale=1.0, kmax=None, n=NV, mass=None):
    """<x^m>_phi, m = 0..mmax, of the 0+ or 1+ diquark leading-twist DA (normalised: <x^0> = 1).
    Returns (moments, raw m = 0 value)."""
    m = MASS[ch] if mass is None else mass
    P = np.array([0, 0, 0, 1j * m])
    k, W = loop4d_full(nq, nz, ny, scale, kmax)
    kp, km = k + P / 2, k - P / 2
    k2 = np.einsum('nm,nm->n', k, k)
    F = calF(k2 / omega2(m))
    gn = slash(n)
    if ch == '0+':
        GC = (C @ (1j * G5) @ Cdag)[None]
        proj = G5 @ gn
    else:                                   # n_nu i gamma_nu C C^dagger
        GC = (1j * gn)[None]
        proj = gn
    t = np.einsum('ab,nbc,ncd,nda->n', proj, Sq(kp), np.broadcast_to(GC, (len(W), 4, 4)), Sq(km)) * F
    x = (kp @ n) / (n @ P)
    ms = np.arange(mmax + 1)
    Mm = (x[None, :] ** ms[:, None]) @ (W * t)
    return Mm / Mm[0], Mm[0]

# ----------------------------------------------------------- reconstruction ----
def _xi_moments_from_x(xm):
    """<xi^j> (xi = 2x - 1) from <x^m>, m = 0..len-1."""
    from math import comb
    N = len(xm)
    return np.array([sum(comb(j, i) * 2**i * (-1)**(j - i) * xm[i] for i in range(j + 1))
                     for j in range(N)])

class DAForm:
    """phi(x) = N [x(1-x)]^(alpha-1/2) [1 + sum_j a_{2j} C_{2j}^alpha(2x-1)], unit normalised."""
    def __init__(self, alpha, *a):
        self.alpha, self.a = alpha, np.array(a, float)
        self.xg, self.wg = self._grid()
        v = self._raw(self.xg)
        self.N = 1 / (self.wg @ v)

    @staticmethod
    def _grid(n=400):
        t, w = np.polynomial.legendre.leggauss(n)
        # x = (1 - cos(pi u))/2 clusters nodes at the end points (integrable end-point powers)
        u = 0.5 * (t + 1)
        x = 0.5 * (1 - np.cos(np.pi * u))
        return x, 0.5 * w * 0.5 * np.pi * np.sin(np.pi * u)

    def _raw(self, x):
        al, xi = self.alpha, 2 * x - 1
        poly = 1 + sum(aj * gegenbauer(2 * (j + 1), al)(xi) for j, aj in enumerate(self.a))
        return (x * (1 - x))**(al - 0.5) * poly

    def __call__(self, x):
        return self.N * self._raw(np.asarray(x, float))

    def xi_moments(self, jmax):
        xi = 2 * self.xg - 1
        v = self(self.xg)
        return np.array([self.wg @ (xi**j * v) for j in range(jmax + 1)])

    def min_value(self):
        return self(self.xg).min()

    def square_moments(self, mmax):
        """<z^m> of q(z) = phi(z)^2 / int phi^2."""
        v = self(self.xg)**2
        v = v / (self.wg @ v)
        return np.array([self.wg @ (self.xg**m * v) for m in range(mmax + 1)])

def fit_da(xm, use=(2, 4, 6)):
    """Fit alpha and a_2 (... a_{2(len(use)-1)}) to the even xi-moments in `use` (from <x^m>_phi):
    as many parameters as moments, i.e. an interpolation of the moments."""
    xim = _xi_moments_from_x(xm)
    npar = len(use)
    def res(p):
        mm = DAForm(*p).xi_moments(max(use))
        return [(mm[j] - xim[j].real) / xim[j].real for j in use]
    best = None
    for a0 in (1.0, 1.5, 2.5, 4.0):
        x0 = [a0] + [0.0] * (npar - 1)
        sol = least_squares(res, x0=x0, bounds=([0.3] + [-1.0] * (npar - 1), [8.0] + [1.0] * (npar - 1)))
        if best is None or sol.cost < best.cost:
            best = sol
    return DAForm(*best.x), xim

def quark_in_diquark(mmax=8, kmax=20.0, **kw):
    """<z^m>_{q/D} for D = 0+, 1+ and the DA fits.  Returns dict ch -> (z_moments, DAForm, info)."""
    out = {}
    for ch in ('0+', '1+'):
        xm, raw = da_moments(ch, mmax=max(mmax, 6), kmax=kmax, **kw)
        d, xim = fit_da(xm.real, use=(2, 4, 6))
        d2, _ = fit_da(xm.real, use=(2, 4))
        out[ch] = dict(z=d.square_moments(mmax), z_2par=d2.square_moments(mmax), da=d, da_2par=d2,
                       x_moments=xm, xi_moments=xim, xi_fit=d.xi_moments(8), raw0=raw)
    return out
