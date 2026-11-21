"""PDF step 3: x-dependence of valence DFs from their Mellin moments (illustrative).

The momentum-dependent Faddeev amplitude gives reliable moments <x^m>, m = 0..9-10 (qdq/pdf.py,
scripts/12).  Reconstructing q(x) from so few moments is ill-posed: positive forms with very
different end-point behaviour (e.g. x^0.4 vs x^2.9 at small x) reproduce them equally well.  We fit
positive-definite forms to the unit-normalised moments m = 1..M,
    'beta':   N x^a (1-x)^b
    'exp':    N x^a (1-x)^b exp(c x + d x^2)
    'exp-b3': same with b = 3 (hadron-scale QCD expectation for the proton valence DFs)
and show the envelope of acceptable fits as a band.  The moments (and quantities computed from them,
including AO evolution, qdq/evolution.py) are the robust output; x <~ 0.1 and x >~ 0.8 are
extrapolations.
"""
import numpy as np
from scipy.optimize import least_squares

def _grid(n=400):
    t, w = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (t + 1)
    x = 0.5 * (1 - np.cos(np.pi * u))                 # clusters nodes at the end points
    return x, 0.25 * np.pi * w * np.sin(np.pi * u)

XG, WG = _grid()
FORMS = {'beta': (2, None), 'exp': (4, None), 'exp-b3': (3, 3.0)}
# default per-moment relative uncertainties, m = 1..10 (grids, DA reconstruction, lam extrapolation)
REL_ERR = np.array([0.005, 0.005, 0.005, 0.01, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05])

def shape(form, p, x):
    x = np.asarray(x, float)
    if form == 'beta':
        a, b = p
        return x**a * (1 - x)**b
    if form == 'exp':
        a, b, c, d = p
    else:
        a, c, d = p
        b = FORMS[form][1]
    return x**a * (1 - x)**b * np.exp(c * x + d * x**2)

class ValenceDF:
    def __init__(self, form, p, number, chi2=None):
        self.form, self.p, self.number, self.chi2 = form, np.asarray(p), number, chi2
        self.N = 1 / (WG @ shape(form, self.p, XG))

    def __call__(self, x):
        return self.number * self.N * shape(self.form, self.p, x)

    def moments(self, mmax):
        f = self(XG)
        return np.array([WG @ (XG**m * f) for m in range(mmax + 1)])

def fit(xm, form, rel_err=REL_ERR):
    """Fit to <x^m>, m = 0..M (m = 0: quark number).  Returns ValenceDF with chi2/dof."""
    xm = np.asarray(xm, float)
    M = len(xm) - 1
    tgt = xm[1:] / xm[0]
    err = rel_err[:M]
    npar = FORMS[form][0]
    def res(p):
        f = shape(form, p, XG)
        raw = np.array([WG @ (XG**m * f) for m in range(M + 1)])
        return (raw[1:] / raw[0] - tgt) / (tgt * err)
    starts = {'beta': [[1, 3], [2.5, 6]],
              'exp': [[1, 3, 0, 0], [2.5, 6, 0, 0], [0, 3, 5, -6], [0.5, 2, 10, -12]],
              'exp-b3': [[1, 0, 0], [0, 5, -6], [0.5, 12, -15]]}[form]
    lo = {'beta': [-0.9, 0.1], 'exp': [-0.9, 0.1, -40, -40], 'exp-b3': [-0.9, -40, -40]}[form]
    hi = {'beta': [6, 15], 'exp': [6, 15, 40, 40], 'exp-b3': [6, 40, 40]}[form]
    best = None
    for p0 in starts:
        s = least_squares(res, p0, bounds=(lo, hi))
        if best is None or s.cost < best.cost:
            best = s
    dof = max(M - npar, 1)
    return ValenceDF(form, best.x, xm[0], chi2=2 * best.cost / dof)

def band(xm, x, forms=('beta', 'exp', 'exp-b3'), chi2_max=2.0, rel_err=REL_ERR):
    """Envelope of x q(x) over the acceptable fits.  Returns (lower, upper, fits)."""
    fits = [fit(xm, f, rel_err) for f in forms]
    ok = [q for q in fits if q.chi2 <= chi2_max] or fits
    curves = np.array([x * q(x) for q in ok])
    return curves.min(0), curves.max(0), fits
