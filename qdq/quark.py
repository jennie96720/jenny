"""Algebraic dressed light-quark propagator, arXiv:2611.19819 App. A.1, Eqs. (A1)-(A4).

S(p) = -i gamma.p sigma_V(p^2) + sigma_S(p^2) = 1/[i gamma.p A(p^2) + B(p^2)].
All functions are entire, so complex p^2 (needed inside bound-state integrals) is fine.
"""
import numpy as np
from .dirac import slash, I4

LAMBDA = 0.566                      # GeV
PARAMS = dict(mbar=0.00897, b0=0.131, b1=2.90, b2=0.603, b3=0.185, eps=1e-4)

def calF(x):
    """F(x) = (1 - e^{-x})/x, with the x -> 0 limit handled."""
    x = np.asarray(x, dtype=complex)
    small = np.abs(x) < 1e-6
    xs = np.where(small, 1.0, x)
    return np.where(small, 1 - x / 2 + x**2 / 6, -np.expm1(-xs) / xs)

def sigmas(p2, mbar=PARAMS['mbar'], b0=PARAMS['b0'], b1=PARAMS['b1'],
           b2=PARAMS['b2'], b3=PARAMS['b3'], eps=PARAMS['eps'], lam=LAMBDA):
    """Return (sigma_V(p2), sigma_S(p2)) in GeV units: sigma_V [GeV^-2], sigma_S [GeV^-1]."""
    x = np.asarray(p2, dtype=complex) / lam**2
    xm = x + mbar**2
    sS = 2 * mbar * calF(2 * xm) + calF(b1 * x) * calF(b3 * x) * (b0 + b2 * calF(eps * x))
    sV = (1 - calF(2 * xm)) / xm
    return sV / lam**2, sS / lam

def AB(p2, **kw):
    sV, sS = sigmas(p2, **kw)
    den = p2 * sV**2 + sS**2
    return sV / den, sS / den

def mass_function(p2, **kw):
    sV, sS = sigmas(p2, **kw)
    return sS / sV                   # M = B/A = sigma_S/sigma_V

def S(p, **kw):
    """Dirac-matrix propagator for momenta p of shape (..., 4) -> (..., 4, 4)."""
    p = np.asarray(p, dtype=complex)
    p2 = np.einsum('...m,...m->...', p, p)
    sV, sS = sigmas(p2, **kw)
    return -1j * slash(p) * sV[..., None, None] + sS[..., None, None] * I4
