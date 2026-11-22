"""Gluon helicity and proton spin decomposition in the AO scheme (EPJC 84 (2024) 739 Sect. 5;
Cheng et al., PLB 844 (2023) 138074).

Light-front (Jaffe-Manohar) decomposition  1/2 = a_I/2 + l_q(zeta) + DeltaG(zeta) + l_g(zeta):
  hadron scale:  DeltaG = l_g = 0,  l_q = 1/2 - a_I/2,  a_I = Delta u + Delta d;
                 flavour split by the Wandzura-Wilczek-type relation (EPJC Eq. 46), whose zeroth moment is
                 l_f = (<x>_f - Delta f)/2;
  evolution (LO, zeta^2 d/dzeta^2, with the AO charge of qdq/evolution.py):
     d a_I = 0;   d DeltaG = (alpha/2pi) [ 2 a_I + (beta_0/2) DeltaG ],  beta_0 = 11 - 2 n_f/3;
     J_q = a_I/2 + l_q and J_g = DeltaG + l_g evolve like the momentum fractions, so
     J_q(zeta) = <x>_quarks(zeta)/2, J_g(zeta) = <x>_g(zeta)/2 (all of J with the quarks at zeta_H);
  measured (anomaly-corrected) quark helicity  a_I^E(zeta) = a_I - n_f alpha(zeta)/(2 pi) DeltaG(zeta),
  n_f = 4 as in the papers.
Ji decomposition (J_q = Delta q/2 + L_q) uses the gravitational form factors J^q(0) instead.
"""
import numpy as np
from scipy.integrate import solve_ivp
from .evolution import AOEvolution

def delta_g(aI, zeta, ev=None, nf=4):
    """DeltaG(zeta) from DeltaG(zeta_H) = 0 (first moment, LO, AO charge)."""
    ev = AOEvolution() if ev is None else ev
    b0 = 11 - 2 * nf / 3
    def rhs(t, y):
        a = ev.alpha(np.exp(t)) / (2 * np.pi)
        return [a * (2 * aI + 0.5 * b0 * y[0])]
    s = solve_ivp(rhs, (np.log(ev.zH**2), np.log(zeta**2)), [0.0], rtol=1e-10, atol=1e-12)
    return s.y[0, -1]

def decomposition(aI, zeta, ev=None, nf=4, xq_total=1.0):
    """Light-front spin budget at zeta.  xq_total: total quark momentum fraction at zeta_H (= 1)."""
    ev = AOEvolution() if ev is None else ev
    o = ev.evolve(1, {'u': xq_total, 'd': 0.0}, zeta)       # only the quark/glue split matters
    xg = o['g']
    Jq, Jg = 0.5 * (xq_total - xg), 0.5 * xg
    dG = delta_g(aI, zeta, ev, nf)
    alpha = ev.alpha(zeta**2)
    return dict(zeta=zeta, aI=aI, DeltaG=dG, l_q=Jq - 0.5 * aI, l_g=Jg - dG, J_q=Jq, J_g=Jg,
                aIE=aI - nf * alpha / (2 * np.pi) * dG, alpha=alpha,
                total=0.5 * aI + (Jq - 0.5 * aI) + dG + (Jg - dG))

def wandzura_wilczek_lf(x_moment, delta):
    """Hadron-scale light-front OAM of flavour f: l_f = (<x>_f - Delta f)/2."""
    return 0.5 * (x_moment - delta)
