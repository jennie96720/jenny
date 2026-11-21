"""PDF step 4: all-orders (AO) evolution of DF Mellin moments from the hadron scale.

Equations of Lu et al., PLB 830 (2022) 137130 (arXiv:2203.00753) Eqs. (3), as used in EPJC 84:739
and arXiv:2505.10663:
    zeta^2 d/dzeta^2 <x^n>_q      = -a gamma_qq^n <x^n>_q                       (valence, per flavour)
    zeta^2 d/dzeta^2 <x^n>_Sig_q  = -a [gamma_qq^n <x^n>_Sig_q + 2 P_qg(zeta) gamma_qg^n(zeta) <x^n>_g]
    zeta^2 d/dzeta^2 <x^n>_g      = -a [sum_q gamma_gq^n <x^n>_Sig_q + gamma_gg^n <x^n>_g]
with a = alpha_1l(zeta^2)/(4 pi), Sig_q = q + qbar (q = u, d, s, c), sea S_q = Sig_q - q_valence,
    alpha_1l(k^2) = gamma_m pi / ln[K^2(k^2)/Lambda^2], K^2 = (a0^2 + a1 k^2 + k^4)/(b0 + k^2),
                    gamma_m = 12/25, Lambda = 0.234 GeV, a0 = 0.104, a1 = 0.0975, b0 = 0.121 GeV^2
                    (process-independent charge; <x>_valence(2 GeV) = 0.483 cf. 0.477 in 2505.10663),
    zeta_H = 0.331 GeV nominal (0.3266 used, see ZETA_H); all DFs except valence vanish at zeta_H,
    threshold  P_qg(zeta) = [1 + tanh((zeta^2 - delta_q^2)/zeta_H^2)]/2, delta_{u,d,s,c} = 0, 0, 0.1, 0.9 GeV,
    Pauli blocking (proton): P_{f<-g}(x) -> P_{f<-g}(x) + sqrt(3)(1 - 2x) g_f/(1 + (zeta/zeta_H - 1)^2),
                    g_d = 0.34 = -g_u (each for f and fbar), g_s = g_c = 0.
Anomalous dimensions gamma^n = -2 int_0^1 x^n P^(0)(x) (so gamma_qq^1 = 32/9).  The gluon virtual term
uses n_f^eff(zeta) = sum_q P_qg(zeta) (momentum conservation).  See AOEvolution for the threshold
treatment and the Pauli-term sign.  The valence equation is independent of all of these choices.
Validation (scratch tests, recorded in README): with SCI hadron-scale valence moments, glue at 2 GeV =
41.2, 2.70, 0.510 % (n = 1, 2, 3) cf. 41.4, 2.67, 0.500 % in 2505.10663 Table 4.
"""
import numpy as np
from scipy.integrate import solve_ivp

CF, CA, TR = 4 / 3, 3.0, 0.5
ZETA_H_NOMINAL = 0.331
ZETA_H = 0.3266        # with alpha_1l below: <x>_val(2 GeV) = 0.477, <x>_val(3.097 GeV) = 0.441, as in the papers
DELTA = {'u': 0.0, 'd': 0.0, 's': 0.1, 'c': 0.9}
# Pauli-blocking strengths as they enter gamma = -2 int x^n P: with the formula taken literally
# (g_d = +0.34) momentum moves OUT of d + dbar; the papers state (and their Table 4 shows) the
# opposite, so the sign is chosen to give dbar > ubar.
PAULI = {'u': 0.34, 'd': -0.34, 's': 0.0, 'c': 0.0}
FLAVOURS = ('u', 'd', 's', 'c')

def alpha_1l(k2, gm=12 / 25, Lam=0.234, a0=0.104, a1=0.0975, b0=0.121):
    """Process-independent effective charge (Cui et al., Chin. Phys. C 44 (2020) 083102):
    gamma_m pi / ln[K^2/Lambda^2], K^2 = (a0^2 + a1 k^2 + k^4)/(b0 + k^2); alpha(0) = 0.98 pi."""
    K2 = (a0**2 + a1 * k2 + k2**2) / (b0 + k2)
    return gm * np.pi / np.log(K2 / Lam**2)

def alpha_simple(k2, gm=12 / 25, Lam=0.234, ma=0.30):
    """Simplified form gamma_m pi / ln[(m_alpha^2 + k^2)/Lambda^2] (gives ~9 % less evolution)."""
    return gm * np.pi / np.log((ma**2 + k2) / Lam**2)

def S1(N):
    return sum(1 / k for k in range(1, N + 1))

def gamma_qq(n):
    N = n + 1
    return CF * (4 * S1(N) - 3 - 2 / (N * (N + 1)))

def gamma_qg(n):
    N = n + 1
    return -2 * TR * (N * N + N + 2) / (N * (N + 1) * (N + 2))

def gamma_gq(n):
    N = n + 1
    return -2 * CF * (N * N + N + 2) / (N * (N * N - 1)) if N > 1 else np.inf

def gamma_gg(n, nf):
    N = n + 1
    return -2 * (2 * CA * (-1 / 12 + 1 / (N * (N - 1)) + 1 / ((N + 1) * (N + 2)) - (S1(N) - 1))
                 - 2 / 3 * nf * TR)

def gamma_pauli(n):
    """-2 int x^n sqrt(3)(1 - 2x) dx (per unit g_f and unit damping factor)."""
    return -2 * np.sqrt(3) * (1 / (n + 1) - 2 / (n + 2))

def threshold(zeta, q, zH=ZETA_H):
    return 0.5 * (1 + np.tanh((zeta**2 - DELTA[q]**2) / zH**2))

class AOEvolution:
    """threshold_mode:
      'share' (default): P_qg^f -> 4 P_qg^f / sum_f P_qg^f, i.e. the thresholds only share the sea
               among flavours; glue and total sea are those of the massless case.  This reproduces the
               papers' glue and total sea (2505.10663 Table 4: 41.4 %, 10.8 % at 2 GeV) and their
               statement that the glue fraction is as in the all-light case.
      'plain': P_qg^f as written, gluon virtual term with n_f^eff = sum_f P_qg^f (momentum conserving).
    Neither reproduces the papers' flavour split of the sea exactly (s too large, light sea too small);
    that split is to be confirmed with the authors.  Valence moments do not depend on any of this."""
    def __init__(self, zeta_H=ZETA_H, pauli=PAULI, threshold_mode='share', alpha=alpha_1l):
        self.zH, self.pauli, self.mode, self.alpha = zeta_H, pauli, threshold_mode, alpha

    def valence_factor(self, n, zeta):
        """<x^n>_q(zeta) / <x^n>_q(zeta_H) for a valence (non-singlet) DF."""
        from scipy.integrate import quad
        I = quad(lambda t: self.alpha(np.exp(t)) / (4 * np.pi), np.log(self.zH**2), np.log(zeta**2))[0]
        return np.exp(-gamma_qq(n) * I)

    def evolve(self, n, valence, zeta):
        """valence: {q: <x^n>_q(zeta_H)} (missing flavours = 0).  Returns dict of moments at zeta:
        valence q, Sigma_q, sea S_q (= Sigma_q - q, i.e. q_sea + qbar), glue g."""
        v0 = np.array([valence.get(q, 0.0) for q in FLAVOURS])
        y0 = np.concatenate([v0, v0, [0.0]])               # valence(4), Sigma(4), g
        gqq, gqg, ggq = gamma_qq(n), gamma_qg(n), gamma_gq(n)
        gpa = gamma_pauli(n)
        zH = self.zH
        def rhs(t, y):
            zeta = np.exp(t / 2)
            a = self.alpha(zeta**2) / (4 * np.pi)
            th = np.array([threshold(zeta, q, zH) for q in FLAVOURS])
            if self.mode == 'share':
                th = th * 4 / th.sum()
            damp = 1 / (1 + (zeta / zH - 1)**2)
            gqg_f = th * gqg + np.array([self.pauli[q] for q in FLAVOURS]) * gpa * damp
            nf = th.sum()
            V, Sg, g = y[:4], y[4:8], y[8]
            dV = -a * gqq * V
            dS = -a * (gqq * Sg + 2 * gqg_f * g)
            dg = -a * (ggq * Sg.sum() + gamma_gg(n, nf) * g)
            return np.concatenate([dV, dS, [dg]])
        sol = solve_ivp(rhs, (np.log(zH**2), np.log(zeta**2)), y0, rtol=1e-10, atol=1e-13)
        y = sol.y[:, -1]
        out = {'g': y[8]}
        for i, q in enumerate(FLAVOURS):
            out[q] = y[i]
            out['Sigma_' + q] = y[4 + i]
            out['S_' + q] = y[4 + i] - y[i]
        return out
