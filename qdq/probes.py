"""External probes coupling to the quark+diquark nucleon: photon (EM current) and graviton (EMT).

A probe supplies, with a flattened probe index m (EM: m = mu; gravity: m = 4 mu + nu):
  quark(l1, l2)   unamputated quark vertex, already including the factor i   -> (N, m, 4, 4)
  dq_s(l1, l2)    unamputated scalar-diquark vertex                          -> (N, m)
  dq_a(l1, l2)    unamputated axial-diquark vertex                           -> (N, m, al, be)
  trans(l1, l2)   amputated 0+ <-> 1+ transition vertex                      -> (N, m, al)
  flavour(charge, phi)                  flavour factors of diagrams 1-3
  lift(vec_mid, weight, scal_mid)       seagull: EM structure -> this probe
  projectors(cur, Pf, Pi, Qv)           dual structures (nb, m, 4, 4) and map to form factors
All vertices obey the probe's Ward-Green-Takahashi identity in the convention of
qdq/current.py (photon: Q.V = G(in) - G(out); graviton: Q_mu V_{mu nu} = l_in,nu G(in) - G(out) l_out,nu).

Graviton vertices: arXiv:2611.19819 Sect. 3 (Eqs. 13-27), i.e. the EM vertex times l_nu, minus
delta_{mu nu} x (mean propagator), plus (quark only) the isoscalar-scalar-meson pole term;
diquark mass radii r_sc = 0.23 fm, r_av = 0.78 fm.  The diquark (and transition) flavour factor is
the AVERAGE over its two quarks (the diquark field carries the momentum of both), i.e. half the
EM-style sum; this is the single-term coefficient of 2611.19819 App. B and is what the EMT Ward
identity requires.
"""
import numpy as np
from .dirac import G, I4, slash
from .quark import S as Sq
from .faddeev import dot
from .current import (Vertices, VERTEX_PARAMS, SIGMA, flavour_factors, lam_plus)

class EMProbe:
    nidx = 4
    name = 'em'

    def __init__(self, mdq=(0.79, 0.89), M=1.18, params=VERTEX_PARAMS):
        self.V = Vertices(mdq, M, params)

    def quark(self, l1, l2):
        return 1j * self.V.chi_quark(l1, l2)

    def dq_s(self, l1, l2):
        return self.V.chi_s(l1, l2)

    def dq_a(self, l1, l2):
        return self.V.chi_a(l1, l2)

    def trans(self, l1, l2):
        return self.V.chi_01(l1, l2)

    def flavour(self, charge, phi):
        return flavour_factors(charge, phi)

    def lift(self, vec_mid, weight, scal_mid):
        return vec_mid

    def projectors(self, M, Pf, Pi, Qv):
        """(F1, F2, F3) via T1 = Lf gamma Li, T2 = Lf sigma.Q Li/2M, T3 = Lf Q Li/2M."""
        Lf, Li = lam_plus(Pf, M), lam_plus(Pi, M)
        sQ = np.einsum('mnab,n->mab', SIGMA, Qv)
        T = [Lf @ G @ Li, Lf @ sQ @ Li / (2 * M), Lf @ (Qv[:, None, None] * I4) @ Li / (2 * M)]
        Tt = [Li @ G @ Lf, Li @ sQ @ Lf / (2 * M), Li @ (Qv[:, None, None] * I4) @ Lf / (2 * M)]
        nb = 1 if np.allclose(Qv, 0) else 3
        Gram = np.array([[np.einsum('mab,mba->', Tt[a], T[b]) for b in range(nb)] for a in range(nb)])
        return np.array(Tt[:nb]), np.linalg.inv(Gram)

GRAV_PARAMS = dict(VERTEX_PARAMS, r_sc=0.23, r_av=0.78, m_S=0.55, r_S=0.193)
GFF_NAMES = ['A', 'J', 'D', 'S', 'cbar']

class GravityProbe:
    nidx = 16
    name = 'grav'

    def __init__(self, mdq=(0.79, 0.89), M=1.18, params=GRAV_PARAMS):
        self.V = Vertices(mdq, M, params)
        self.p = params

    @staticmethod
    def _lnu(l, vec):
        """l_nu vec_mu... -> flattened (N, 16, ...)."""
        out = vec[:, :, None] * l.reshape(l.shape[0], 1, 4, *([1] * (vec.ndim - 2)))
        return out.reshape(vec.shape[0], 16, *vec.shape[2:])

    @staticmethod
    def _delta(x):
        """delta_{mu nu} x -> (N, 16, ...)."""
        d = np.eye(4).reshape(16)
        return d.reshape(1, 16, *([1] * (x.ndim - 1))) * x[:, None]

    def quark(self, l1, l2):
        """i chi^q_{mu nu}: 2611.19819 Eq. (14) with the scalar-pole transverse term, Eq. (19)."""
        V, p = self.V, self.p
        l, Q = 0.5 * (l1 + l2), l1 - l2
        S1, S2 = Sq(l1), Sq(l2)
        out = self._lnu(l, 1j * V.chi_quark(l1, l2)) - 0.5 * self._delta(S1 + S2)
        Q2 = dot(Q, Q)
        TP = -p['r_S'] * (Q2[:, None, None] * np.eye(4) - Q[:, :, None] * Q[:, None, :]) \
             / (Q2 + p['m_S']**2)[:, None, None]                      # T_{mu nu}(Q) P^S(Q^2)
        out = out - TP.reshape(-1, 16)[:, :, None, None] * (S1 @ S2)[:, None]
        return out

    def dq_s(self, l1, l2):
        V = self.V
        lc = l1 - l2
        F = V.Fj(dot(lc, lc), self.p['r_sc'])
        sig = 0.5 * (V.D0(dot(l1, l1)) + V.D0(dot(l2, l2)))
        return self._lnu(0.5 * (l1 + l2), V.chi_s(l1, l2)) - self._delta(sig * F)

    def dq_a(self, l1, l2):
        V = self.V
        lc = l1 - l2
        F = V.Fj(dot(lc, lc), self.p['r_av'])
        sig = 0.5 * (V.D1mat(l1) + V.D1mat(l2)) * F[:, None, None]
        return self._lnu(0.5 * (l1 + l2), V.chi_a(l1, l2)) - self._delta(sig)

    def trans(self, l1, l2):
        return self._lnu(0.5 * (l1 + l2), self.V.chi_01(l1, l2))

    def flavour(self, charge, phi):
        f1, f2, f3 = flavour_factors(charge, phi)
        return f1, {k: v / 2 for k, v in f2.items()}, {k: v / 2 for k, v in f3.items()}

    def lift(self, vec_mid, weight, scal_mid):
        """Seagull: weight_nu x (EM structure)_mu + delta_{mu nu}/2 x Gamma (2611.19819 Eq. 26)."""
        N = vec_mid.shape[0]
        w = np.broadcast_to(weight, (N, 4))
        out = vec_mid[:, :, None] * w[:, None, :, None, None]           # (N, mu, nu, 4, 4)
        return out.reshape(N, 16, 4, 4) + 0.5 * self._delta(scal_mid)

    def projectors(self, M, Pf, Pi, Qv):
        """s_1..s_5 of 2611.19819 Eq. (5) and the linear map to (A, J, D, S, cbar), Eq. (6).
        The map includes the phase relating our raw current (divided by the baryon-number
        normalisation B) to Lambda_{mu nu}: Lambda = -i J_raw / B.  |phase| = 1 follows from the
        EM normalisation (B = -i N^2 x baryon charge); its sign is fixed by A(0) = +1, so the
        magnitude |A(0)| = 1 (momentum sum rule) remains a genuine check."""
        P = 0.5 * (Pf + Pi)
        P2 = dot(P, P)
        Q2 = dot(Qv, Qv)
        gQ = slash(Qv)
        # gamma^perp_mu = gamma_mu - Q_mu gamma.Q / Q^2; at Q = 0 take the limit along the Breit-frame
        # direction of Q (axis 3), otherwise s_3 = s_4 identically and A, cbar are mixed up
        Qh = Qv / np.sqrt(Q2) if abs(Q2) > 0 else np.array([0, 0, 1, 0], complex)
        gp = G - Qh[:, None, None] * slash(Qh)[None]
        PP = P[:, None] * P[None, :]
        D = np.zeros((5, 4, 4, 4, 4), complex)
        D[0] = (M / P2**2) * PP[:, :, None, None] * I4
        D[1] = (1j / P2) * gp[:, None] * P[None, :, None, None]
        D[2] = (M / P2) * np.eye(4)[:, :, None, None] * I4
        D[3] = (M / P2) * np.einsum('mab,nbc->mnac', gp, gp)
        D[4] = (1j / P2) * P[:, None, None, None] * gp[None]
        Tt = D.reshape(5, 16, 4, 4)
        tau = (Q2 / (4 * M**2)).real
        if tau == 0:
            Map = np.array([[-5, -1, 0, 1, -1], [0, 0, -0.5, 0.5, 0]]) / np.array([[4], [1]])
            Map = Map.astype(complex)             # rows: A, cbar  (J, D, S need Q != 0)
        else:
            Map = np.array([
                [-5, -1, 0, 1, -1],
                [-2, -1, 0, 0, -1],
                [1, 0, 2, -3, 0],
                [0, -1, 0, 0, 1],
                [0, 0, -0.5, 0.5, 0]], complex) / np.array([[4 * (1 + tau)], [8 * tau], [4 * tau], [8 * tau], [1]])
        return Tt, -1j * Map

def emt_matrix_element(M, Pf, Pi, A, J, D, S, cbar):
    """Lambda_{mu nu} = -Lf [P P A + i P_{(mu} sigma_{nu) rho} Q J + (QQ - delta Q^2) D/4
    - i P_{[mu} sigma_{nu] rho} Q S - delta M^2 cbar] Li / M   (2611.19819 Eqs. 1, 4); for tests."""
    P, Q = 0.5 * (Pf + Pi), Pf - Pi
    Lf, Li = lam_plus(Pf, M), lam_plus(Pi, M)
    sQ = np.einsum('mnab,n->mab', SIGMA, Q)                             # sigma_{mu rho} Q_rho
    Psq = P[:, None, None, None] * sQ[None]                             # P_mu sigma_{nu rho} Q_rho
    sym, asym = 0.5 * (Psq + Psq.transpose(1, 0, 2, 3)), 0.5 * (Psq - Psq.transpose(1, 0, 2, 3))
    d = np.eye(4)
    X = (P[:, None] * P[None, :] * A)[:, :, None, None] * I4 + 1j * J * sym \
        + (0.25 * D * (Q[:, None] * Q[None, :] - d * dot(Q, Q)))[:, :, None, None] * I4 \
        - 1j * S * asym - (d * M**2 * cbar)[:, :, None, None] * I4
    return -np.einsum('ab,mnbc,cd->mnad', Lf, X, Li) / M
