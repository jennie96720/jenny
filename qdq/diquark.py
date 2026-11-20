"""Diquark correlation amplitudes, propagators and canonical normalisation.
arXiv:2611.19819 App. A.2-A.3, Eqs. (A5)-(A11).

  Gamma^{0+}(k;K)      = g0 gamma_5 C t^0 H_c F(k^2/w0^2)
  Gamma^{1+}_mu(k;K)   = g1 i gamma_mu C t^j H_c F(k^2/w1^2)
  Gammabar(k;K)        = C^dagger Gamma(-k;K)^T C
  Delta^{0+}(K)        = Pi^{0+}(K^2),  Delta^{1+}_{mu nu} = [delta + K K/m^2] Pi^{1+}(K^2)
  Pi^{JP}(K^2)         = F(K^2/w^2)/m^2,   w^2 = m^2/2.
"""
import numpy as np
from .dirac import G, G5, C, Cdag, I4
from .quark import S, calF
from .quadrature import loop4d

MASS = {'0+': 0.79, '1+': 0.89}     # GeV, Eq. (A9)
def omega2(m):                       # Eq. (A11)
    return 0.5 * m**2

# ---- colour-flavour traces --------------------------------------------------
# Gell-Mann matrices needed for H_c = {i l7, -i l5, i l2}
l2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
l5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
l7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
H = [1j * l7, -1j * l5, 1j * l2]
tau = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
       np.array([[1, 0], [0, -1]])]
t_f = [1j / np.sqrt(2) * tau[2], 0.5 * (tau[0] + tau[3]), tau[1] / np.sqrt(2),
       0.5 * (tau[0] - tau[3])]

def cf_trace(tf, h):
    """Tr_CF[(t h)^T (t h)] for one diquark of definite flavour and colour component."""
    return np.trace(tf.T @ tf) * np.trace(h.T @ h)

# ---- Dirac structures (coupling and F-factor stripped) ------------------
def dirac_amp(ch, mu=None):
    # Alkofer et al. nucl-th/0412046 Eq. (15): C i gamma_5 and i gamma_mu C.
    # (2611.19819 Eq. (A5) writes gamma_5 C, which differs by a factor i.)
    return C @ (1j * G5) if ch == '0+' else 1j * G[mu] @ C

def dirac_bar(ch, mu=None):
    return Cdag @ dirac_amp(ch, mu).T @ C

# ---- one-loop "self-energy" of the diquark --------------------------------
def Pi_loop(ch, Q, K, m, nq=160, nz=24, Gamma_sq=None):
    """Tr_D int d^4q/(2pi)^4 Gbar(q;-K) S(q+Q/2) G(q;K) S^T(-q+Q/2) F^2, coupling g = 1.
    For 1+ returns the Lorentz-index matrix Pi_{alpha beta}. Colour-flavour trace NOT included."""
    qv, W = loop4d(nq, nz, scale=1.0)
    w2 = omega2(m)
    q2 = np.einsum('...m,...m->...', qv, qv)
    F2 = calF(q2 / w2) ** 2
    Sp = S(qv + Q / 2)
    SmT = np.swapaxes(S(-qv + Q / 2), -1, -2)
    if ch == '0+':
        val = np.einsum('ab,...bc,cd,...da->...', dirac_bar('0+'), Sp, dirac_amp('0+'), SmT)
        return np.sum(W * F2 * val)
    out = np.zeros((4, 4), dtype=complex)
    for a in range(4):
        for b in range(4):
            val = np.einsum('ab,...bc,cd,...da->...', dirac_bar('1+', a), Sp,
                            dirac_amp('1+', b), SmT)
            out[a, b] = np.sum(W * F2 * val)
    return out

def coupling(ch, nq=160, nz=24, h=1e-3, cf=None):
    """Solve the canonical normalisation, Eq. (A8), for g_{J^P}.
    Contract both sides with K_mu (and T_{alpha beta} for 1+):
        0+:  2 K^2      = d/da Pi(a K)|_{a=1}
        1+:  2 K^2 * 3  = T_{ab} d/da Pi_{ab}(a K)|_{a=1}
    with K = (0,0,0, i m) in the rest frame, so K^2 = -m^2."""
    m = MASS[ch]
    K = np.array([0, 0, 0, 1j * m])
    if cf is None:
        cf = cf_trace(t_f[0] if ch == '0+' else t_f[1], H[0])
    if ch == '0+':
        f = lambda a: Pi_loop(ch, a * K, K, m, nq, nz)
        lhs = 2 * (-m**2)
    else:
        T = np.eye(4) - np.outer(K, K) / (-m**2)
        f = lambda a: np.sum(T * Pi_loop(ch, a * K, K, m, nq, nz))
        lhs = 2 * (-m**2) * 3
    dPi = (f(1 + h) - f(1 - h)) / (2 * h) * cf
    g2 = lhs / dPi
    # complex sqrt: with these gamma-matrix conventions g_{1+}^2 < 0, i.e. g_{1+} is imaginary;
    # the phase must be kept because g_X g_Y (not |g|^2) enters the Faddeev kernel.
    return np.sqrt(complex(g2.real)), g2, cf
