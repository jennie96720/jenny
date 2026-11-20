"""Quadrature rules for 4D Euclidean loop integrals."""
import numpy as np

def radial(n, scale=1.0):
    """Gauss-Legendre on t in (0,1) mapped to q = scale * t/(1-t) in (0, inf).
    Returns nodes q and weights w such that sum w f(q) ~ int_0^inf dq f(q)."""
    t, wt = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (t + 1); wt = 0.5 * wt
    q = scale * t / (1 - t)
    return q, wt * scale / (1 - t)**2

def chebyshev_u(n):
    """Gauss-Chebyshev (2nd kind): sum w f(z) ~ int_{-1}^{1} dz sqrt(1-z^2) f(z)."""
    k = np.arange(1, n + 1)
    z = np.cos(k * np.pi / (n + 1))
    w = np.pi / (n + 1) * np.sin(k * np.pi / (n + 1))**2
    return z, w

def loop4d(nq=200, nz=32, scale=1.0):
    """Nodes/weights for int d^4q/(2pi)^4 f(q) when f depends on q only via (|q|, q4 = |q| z).
    Uses d^4q = q^3 dq * 4pi sqrt(1-z^2) dz  (the two remaining angles integrate trivially).
    Returns q4-vectors of shape (nq, nz, 4) with q along (0, 0, |q| sqrt(1-z^2), |q| z)."""
    q, wq = radial(nq, scale)
    z, wz = chebyshev_u(nz)
    Q, Zz = np.meshgrid(q, z, indexing='ij')
    W = (wq[:, None] * q[:, None]**3) * wz[None, :] * 4 * np.pi / (2 * np.pi)**4
    vec = np.zeros(Q.shape + (4,))
    vec[..., 2] = Q * np.sqrt(1 - Zz**2)
    vec[..., 3] = Q * Zz
    return vec, W
